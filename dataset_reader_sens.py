from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import re


STATE_LABELS = ['s', 'n', 'mu', 'vx', 'vy', 'r']


def extract_track_width_from_filename(filename):
    name = Path(filename).name
    match = re.search(r"_tw(?P<track_width>\d+(?:\.\d+)?)_", name)
    if match is None:
        raise ValueError(
            f"Could not extract track width from filename '{name}'. "
            "Expected pattern like '..._tw0.3_....npz'."
        )
    return float(match.group("track_width"))
    

class F1tenthSensDataset(torch.utils.data.Dataset):

    def __init__(self,
                 list_of_npz_paths : str,
                 rollout_length: int = 40,
                 skip: int = 1,
                 sensitivities_strategy: str = 'ones', # 'sens', 'ones', 'stand', 'norm', 'mpc_cost' 'sens_norm'
                 states_loss_strategy: str = 'all', # 'pose_only', 'v_only', 'all'
                 clamp_percentile: float = 95.44, # 4 std dev for sensitivity clamping, 3 std dev is 98.76 percentile, 2 std dev is 95.44 percentile
                 split_val_tws: list = [0.35],
                 split = 'train'
                 ):
        
        
        self.nu = 2
        self.nx = 6
        self.N = 80 # from MPC
        self.rollout_length = rollout_length
        self.skip = skip    
        self.sensitivities_strategy = sensitivities_strategy
        self.state_loss_strategy = states_loss_strategy
        
        assert rollout_length <= self.N, "Rollout length should be less than or equal to N"
        assert sensitivities_strategy in ['sens', 'ones', 'stand', 'norm', 'mpc_cost', 'sens_stand'], "Invalid sensitivities strategy"
        assert states_loss_strategy in ['all', 'pose_only', 'v_only'], "Invalid states loss strategy"
        
        if states_loss_strategy == 'pose_only':
            # only keep sensitivities for pose states (s, n, mu)
            self.state_slice = slice(0, 6)
        elif states_loss_strategy == 'v_only':
            # only keep sensitivities for velocity states (vx, vy, r)
            self.state_slice = slice(3, 6)
        else:
            self.state_slice = slice(0, 6)
        
        x_list = []
        u_list = []
        sens_list = []    
        sens_valid_list = []    
        
        for i in range(len(list_of_npz_paths)):
            file_name_path = list_of_npz_paths[i]

            tw = extract_track_width_from_filename(file_name_path.stem)

            data = np.load(file_name_path, allow_pickle=True)

            x0 = data['x0'][:, :6] # (T, nx)
            u0 = data['x0'][:, 6:] # (T, nu)
            sens = data['sens_J_wrt_e'][:, :6, :] # (T, nx, rollout_length)
            valid_sens = data['sens_valid'] # (T,)

            if split == 'train':
                if tw in split_val_tws:
                    continue
            elif split == 'val':
                if tw not in split_val_tws:
                    continue
            else:
                raise ValueError(f"Invalid split: {split}. Expected 'train' or 'val'.")
            
            x_list.append(x0)
            u_list.append(u0)
            sens_list.append(sens)
            sens_valid_list.append(valid_sens)

        # concatonate on new epispode dimension
        self.x = np.stack(x_list, axis=0) # (num_episodes, T, nx)
        self.u = np.stack(u_list, axis=0) # (num_episodes, T, nu)
        self.sens = np.stack(sens_list, axis=0) # (num_episodes, T, nx, rollout_length+1)
        self.sens = self.sens[:, :, :, :rollout_length] # trim to rollout length
        self.sens_valid = np.stack(sens_valid_list, axis=0) # (num_episodes, T)
        
        # Compute statistics for standardization and normalization
        self.x_std = np.std(self.x, axis=(0, 1))
        self.x_max = np.max(self.x, axis=(0, 1))
        self.x_min = np.min(self.x, axis=(0, 1))
        
        dx_dt = np.diff(self.x, axis=1)
        dx_dt[:, :, 0] = np.clip(dx_dt[:, :, 0], -0.5, 0.5) # clamp s dot to filter out outliers that mess up std
        self.dx_dt_std = np.std(dx_dt, axis=(0, 1))
        print(f"State derivative statistics: std={self.dx_dt_std}")
        
        
        if self.sensitivities_strategy in ['sens', 'sens_stand']:
            self.sens_dJ_de_all = np.abs(self.sens).transpose(0, 1, 3, 2) # [num_episodes, T, N+1, nx]
            
            self.mean_per_channel = np.mean(self.sens_dJ_de_all, axis=(0, 1, 2)) # [nx] 
            
            self.max_clamp = np.percentile(self.sens_dJ_de_all, clamp_percentile, axis=(0, 1))
            self.min_clamp = np.percentile(self.sens_dJ_de_all, (100 - clamp_percentile), axis=(0, 1))
            self.sens_dJ_de_all = np.clip(self.sens_dJ_de_all, self.min_clamp, self.max_clamp)
            
            if self.sensitivities_strategy == 'sens_stand':
                self.sens_dJ_de_all = self.sens_dJ_de_all / self.x_std
            
            if self.state_loss_strategy == 'pose_only':
                self.sens_dJ_de_all[:, :, :, 3:] = 0.0
            elif self.state_loss_strategy == 'v_only':
                self.sens_dJ_de_all[:, :, :, :3] = 0.0
            
            print(f"Sensitivities stats after processing: mean={np.mean(self.sens_dJ_de_all)}, std={np.std(self.sens_dJ_de_all)}, min={np.min(self.sens_dJ_de_all)}, max={np.max(self.sens_dJ_de_all)}")   
            self.sens_dJ_de_all = self.sens_dJ_de_all / np.mean(self.sens_dJ_de_all) 
                    
                    
        self.sens_val = np.ones((1, self.nx))
        
        # Sensitivity strategy
        if self.sensitivities_strategy == 'ones':
            pass
        elif self.sensitivities_strategy == 'stand':
            self.sens_val = self.sens_val / self.x_std
        elif self.sensitivities_strategy == 'norm':
            self.sens_val = self.sens_val / (self.x_max - self.x_min)
        elif self.sensitivities_strategy == 'mpc_cost':
            self.sens_val = np.ones((1, self.nx))
            
        # State loss strategy
        if self.state_loss_strategy == 'pose_only':
            self.sens_val[0, 3:] = 0.0
        elif self.state_loss_strategy == 'v_only':
            self.sens_val[0, :3] = 0.0
            
        self.sens_val = self.sens_val / np.mean(self.sens_val)
                
        list_of_start_indices = []
        
        for i_episode in range(self.x.shape[0]):
            for i_timestep in range(0, self.x.shape[1] - rollout_length, skip):
                # check if sens is valid for this time step
                if self.sens_valid[i_episode, i_timestep] == 0:
                    continue
                
                # check vx > 1.4 m/s to filter out low-speed data
                if np.any(self.x[i_episode, i_timestep:, 3] < 0.64):
                    continue
                
                list_of_start_indices.append((i_episode, i_timestep))
        
        self.idxs = np.stack(list_of_start_indices)
        print(self.idxs.shape)
            
    def __len__(self):
        return self.idxs.shape[0]
    
    def __getitem__(self, idx: np.ndarray):
        
        # Handle single integer index
        if isinstance(idx, int):
            idx = np.array([idx], dtype=np.int64)
            
        episodes_idx_and_time_start_idx = self.idxs[idx]
        episodes_idx = episodes_idx_and_time_start_idx[:, 0]
        time_start_idx = episodes_idx_and_time_start_idx[:, 1]

        seq_indices = time_start_idx[:, None] + np.arange(self.rollout_length + 1)
        
        x_seq = self.x[episodes_idx[:, None], seq_indices, :] # (batch_size, rollout_length+1, nx)
        u_seq = self.u[episodes_idx[:, None], seq_indices, :] # (batch_size, rollout_length+1, nu)
        
        
        if self.sensitivities_strategy in ['sens', 'sens_stand']:    
            sens_batch = self.sens_dJ_de_all[episodes_idx[:, None], time_start_idx[:, None]][:, 0, :, :] # (batch_size, rollout_length, nx)
            init_time_step_sens = np.zeros_like(sens_batch[:, 0, :]) # (batch_size, nx)
            sens_batch = np.concatenate([init_time_step_sens[:, None, :], sens_batch[:, :, :]], axis=1) # (batch_size, rollout_length+1, nx)
        else:
            sens_batch = np.tile(self.sens_val, (episodes_idx.shape[0], self.rollout_length+1, 1)) # (batch_size, rollout_length+1, nx)    
        
        x0 = x_seq[:, 0, self.state_slice] # (batch_size, nx)
        x_seq = x_seq[:, :, self.state_slice]
        sens_batch = sens_batch[:, :, self.state_slice]

        x0 = torch.from_numpy(x0).float()
        x0 = x0.unsqueeze(1) # (batch_size, 1, nx)
        u_seq = torch.from_numpy(u_seq).float()
        x_seq = torch.from_numpy(x_seq).float()
        sens_batch = torch.from_numpy(sens_batch).float()
        

        return x0, u_seq, x_seq, sens_batch, [episodes_idx, time_start_idx]
    
    def plot_mean_sensitivities(self):
        if not self.sensitivities_strategy in ['sens', 'sens_stand']:
            print("Mean sensitivities plot is only available for 'sens' strategy.")
            return
        
        # plot mean sensitivities over all episodes and time steps
        # plot std sensitivities over all episodes and time steps
        mean_sens = np.mean(self.sens_dJ_de_all, axis=(0, 1))  # [N+1, nx]
        std_sens = np.std(self.sens_dJ_de_all, axis=(0, 1))    # [N+1, nx]
        
        # percentile = 99.5
        # max_clamp = np.percentile(self.sens_dJ_de_all, percentile, axis=(0, 1))
        # min_clamp = np.percentile(self.sens_dJ_de_all, 100 - percentile, axis=(0, 1))
        
        max_sens = np.max(self.sens_dJ_de_all, axis=(0, 1))    # [N+1, nx]
        min_sens = np.min(self.sens_dJ_de_all, axis=(0, 1))    # [N+1, nx]
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(mean_sens.T, aspect='auto')
        plt.colorbar()
        plt.title('Mean Sensitivities over all episodes and time steps')
        plt.xlabel('Time Step')
        plt.ylabel('State')
        plt.yticks(range(self.nx), STATE_LABELS)
        plt.subplot(2, 2, 2)
        plt.imshow(std_sens.T, aspect='auto')
        plt.colorbar()
        plt.title('STD of Sensitivities over all episodes and time steps')
        plt.xlabel('Time Step')
        plt.ylabel('State')
        plt.yticks(range(self.nx), STATE_LABELS)
        plt.subplot(2, 2, 3)
        plt.imshow(min_sens.T, aspect='auto')
        plt.colorbar()
        plt.title('Min Sensitivities over all episodes and time steps')
        plt.xlabel('Time Step')
        plt.ylabel('State')
        plt.yticks(range(self.nx), STATE_LABELS)
        plt.subplot(2, 2, 4)
        plt.imshow(max_sens.T, aspect='auto')
        plt.colorbar()
        plt.title(f'Max Sensitivities over all episodes and time steps')
        plt.xlabel('Time Step')
        plt.ylabel('State')
        plt.yticks(range(self.nx), STATE_LABELS)
        plt.tight_layout()
        plt.show()
        


if __name__ == "__main__":

    log_files = Path("./f1enth_long_track_sens")

    paths_list = list(log_files.glob("*.npz"))
    print(f"Found {len(paths_list)} log files for F1tenth dataset.")

    sensivity_strategy = 'sens' # 'sens', 'ones', 'stand', 'norm', 'mpc_cost'
    states_loss_strategy = 'pose_only' # 'pose_only', 'v_only', 'all'

    dataset = F1tenthSensDataset(paths_list, sensitivities_strategy=sensivity_strategy,
                            states_loss_strategy=states_loss_strategy,
                            split='train', skip=3)

    dataset.plot_mean_sensitivities()   

    batch_size = 32
    sampler = torch.utils.data.BatchSampler(
        torch.utils.data.SequentialSampler(dataset),
        batch_size=batch_size,
        drop_last=True
    )
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=None,
                                                sampler=sampler, num_workers=0)
        
    import matplotlib.animation as animation
    from matplotlib.lines import Line2D
    import matplotlib.colors as colors

    strategy = dataset.sensitivities_strategy
    states_loss_strategy = dataset.state_loss_strategy
    rollout_length = dataset.rollout_length

    plot_type = 'all'  # Options: 'all', 'pose_only', 'v_only'
    sens_chart_filter = 'all'  # Options: 'all', 'pose_only', 'v_only'
    plot_scale = 'linear'  # Options: 'linear' or 'log'

    drop_s_in_time_series = False  # Whether to drop the 's' state from the time series plot for better visibility of other states
    scale_s_in_time_series = 1/15  # Whether to scale the 's' state in the time series plot to fit better with other states
    plot_controls = True  # Whether to overlay control signals on the time series plot

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 3]})
    joint_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    all_labels = STATE_LABELS
    state_indices = np.arange(dataset.nx)[dataset.state_slice]
    selected_labels = [all_labels[i] for i in state_indices]
    selected_set = set(state_indices.tolist())

    if sens_chart_filter == 'pose_only':
        active_indices = [i for i in state_indices if i < 3]
    elif sens_chart_filter == 'v_only':
        active_indices = [i for i in state_indices if i >= 3]
    else:
        active_indices = state_indices.tolist()

    if len(active_indices) == 0:
        active_indices = state_indices.tolist()

    sens_dims = len(active_indices)
    y_labels = [all_labels[i] for i in active_indices]
    sens_plot_indices = [selected_labels.index(all_labels[i]) for i in active_indices]

    # Calculate vmin and vmax for the sensitivity plot globally
    if strategy in ['sens', 'sens_stand']:
        global_vmax = np.percentile(dataset.sens_dJ_de_all[:, :, :, active_indices], 95) # keep maximum, or can be 95 percentile
        global_vmin = np.percentile(dataset.sens_dJ_de_all[:, :, :, active_indices], 5) # keep minimum, or can be 5 percentile
    else:
        global_vmax = np.max(dataset.sens_val[:, active_indices])
        global_vmin = np.min(dataset.sens_val[:, active_indices])

    vmin = global_vmin
    vmax = global_vmax
    colorsbar = colors.SymLogNorm(linthresh=0.01, linscale=0.5, vmin=vmin, vmax=vmax) if plot_scale == 'log' else colors.Normalize(vmin=vmin, vmax=vmax)

    im = ax1.imshow(np.zeros((sens_dims, rollout_length + 1)), aspect='auto', cmap='viridis', norm=colorsbar)
    plt.colorbar(im, ax=ax1)

    ax1.set(title=f'{strategy} {states_loss_strategy} - Showing: {sens_chart_filter}',
            xlabel='Time Step', ylabel='State Dimension', yticks=range(sens_dims), yticklabels=y_labels)

    for i, tick_label in enumerate(ax1.get_yticklabels()):
        # Get the correct color based on the label, to keep colors consistent
        label_idx = all_labels.index(tick_label.get_text())
        tick_label.set_color(joint_colors[label_idx % len(joint_colors)])

    if plot_type == 'pose_only':
        plot_indices = [i for i in state_indices if i < 3]
        plot_type_label = 'Pose (s, n, mu)'
    elif plot_type == 'v_only':
        plot_indices = [i for i in state_indices if i >= 3]
        plot_type_label = 'Velocities (vx, vy, r)'
    else:
        plot_indices = state_indices.tolist()
        plot_type_label = 'All States'

    if len(plot_indices) == 0:
        plot_indices = state_indices.tolist()

    plot_dims = len(plot_indices)
    plot_labels = [all_labels[i] for i in plot_indices]

    if drop_s_in_time_series and 's' in plot_labels:
        plot_indices = [i for i in plot_indices if all_labels[i] != 's']
        plot_labels = [all_labels[i] for i in plot_indices]
        plot_dims = len(plot_indices)
        if plot_dims == 0:
            plot_indices = state_indices.tolist()
            plot_labels = [all_labels[i] for i in plot_indices]
            plot_dims = len(plot_indices)

    plot_series_indices = [selected_labels.index(label) for label in plot_labels]
        
    line_styles = [(0.3, '-'), (1, '-')]
    lines = [[ax2.plot([], [], color=joint_colors[all_labels.index(plot_labels[i])], alpha=alpha, linestyle=ls, linewidth=2 if alpha==1 else 1)[0] 
            for i in range(plot_dims)] for alpha, ls in line_styles]
    lines_q_full, lines_q_sample = lines

    plot_type_label = 'States'
    state_legend_elements = [Line2D([0], [0], color=joint_colors[all_labels.index(label)], lw=2, label=label) 
                            for label in plot_labels]
    legend_elements = [Line2D([0], [0], color='gray', alpha=0.3, label='Full episode'),
                    Line2D([0], [0], color='gray', linewidth=2, label='Current sample')] + state_legend_elements

    all_lines = lines_q_full + lines_q_sample

    if plot_controls:
        control_labels = ['wheel_speed', 'delta']
        control_colors = ['#17becf', '#bcbd22']
        lines_u = [[ax2.plot([], [], color=control_colors[i], alpha=alpha, linestyle='--', linewidth=2 if alpha==1 else 1)[0]
                    for i in range(2)] for alpha, _ in line_styles]
        lines_u_full, lines_u_sample = lines_u
        all_lines += lines_u_full + lines_u_sample
        
        control_legend_elements = [Line2D([0], [0], color=control_colors[i], lw=2, linestyle='--', label=label) 
                                for i, label in enumerate(control_labels)]
        legend_elements += control_legend_elements

    line_plot_labels = ', '.join(plot_labels)
    ax2.set(xlabel='Time Step', ylabel='State Value',
        title=f'Episode Overview - Highlighting Current Sample ({plot_type_label}: [{line_plot_labels}])')
    ax2.legend(handles=legend_elements, loc='upper right')
    ax2.grid(True, alpha=0.3)

    data_iter = iter(dataset_loader)

    def flatten_batches(iterator):
        for batch in iterator:
            x0, u_seq, x_seq, sens_batch, idxs = batch
            for i in range(sens_batch.shape[0]):
                yield sens_batch[i], x_seq[i], u_seq[i], [idxs[0][i], idxs[1][i]]

    flat_data_iter = flatten_batches(data_iter)

    def update(frame):
        try:
            sens_sample, x_seq_sample, u_seq_sample, idxs = next(flat_data_iter)
            ep_idx, t_start = int(idxs[0]), int(idxs[1])
            
            # Update sensitivities heatmap
            data_heatmap = sens_sample.numpy().T[sens_plot_indices, :]
            im.set_data(data_heatmap)
            im.set_clim(vmin=global_vmin, vmax=global_vmax)
            ax1.set_title(f'{strategy} {states_loss_strategy} - Showing: {sens_chart_filter} - Sample {frame}, e:{ep_idx}, t:{t_start}')
            
            # Update time series plot
            q_full = dataset.x[ep_idx][:, plot_indices].copy()
            q_sample = x_seq_sample.numpy()[:, plot_series_indices].copy()
            
            # Scale 's' if it's included in the plot
            if 's' in plot_labels and scale_s_in_time_series != 1.0:
                s_idx = plot_labels.index('s')
                q_full[:, s_idx] *= scale_s_in_time_series
                q_sample[:, s_idx] *= scale_s_in_time_series
            
            t_full = np.arange(q_full.shape[0])
            t_sample = np.arange(t_start, t_start + q_sample.shape[0])
            
            # Update all lines
            for i, (line_full, line_sample) in enumerate(zip(lines_q_full, lines_q_sample)):
                line_full.set_data(t_full, q_full[:, i])
                line_sample.set_data(t_sample, q_sample[:, i])
                
            y_min = min(q_full.min(), q_sample.min())
            y_max = max(q_full.max(), q_sample.max())
                
            if plot_controls:
                u_full = dataset.u[ep_idx]
                u_sample = u_seq_sample.numpy()
                
                for i, (line_full, line_sample) in enumerate(zip(lines_u_full, lines_u_sample)):
                    line_full.set_data(t_full, u_full[:, i])
                    line_sample.set_data(t_sample, u_sample[:, i])
                    
                y_min = min(y_min, u_full.min(), u_sample.min())
                y_max = max(y_max, u_full.max(), u_sample.max())
            
            # Adjust axis limits
            ax2.set_xlim(0, len(t_full))
            ax2.set_ylim(y_min - 0.1, y_max + 0.1)
            
            return [im] + all_lines
        except StopIteration:
            return [im] + all_lines

    ani = animation.FuncAnimation(fig, update, frames=50000, interval=50, blit=False)

    plt.tight_layout()
    plt.show()