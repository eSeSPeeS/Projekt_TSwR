# Projekt_TSwR

Model Predictive Control dla symulowanego samochodu

Cele projektu: 
- Kamień milowy 1: Stworzenie symulatora samochodu w oparciu o model dynamiczny opisany w publikacji naukowej arXiv:2003.04882
- Kamień milowy 2: Zaimplementowanie regulatora typu Model Predictive Control (MPC), który umożliwi śledzenie zadanych trajektorii ruchu pojazdu.
- Kamień milowy 3 (cel główny): Rozszerzenie regulatora MPC o komponent uczący się, wykorzystujący Gaussian Process Regression do modelowania dynamiki błędu pojazdu

Projekt zostanie zrealizowany w języku Python z wykorzystaniem:
- numpy – obliczenia numeryczne  
- matplotlib – wizualizacja wyników  
- acados – implementacja regulatora MPC  
- scipy – narzędzia do optymalizacji i równań różniczkowych
- pybullet - symulacja samochodu


