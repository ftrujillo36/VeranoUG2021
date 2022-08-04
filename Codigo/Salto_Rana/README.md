# SFLA for Robot Path Planning
This is an implementation of the Shuffled Frog leaping Algorithm for
autonomus navigation of mobile Robots.
It uses the Coppeliasim legacy remote API to control a Pioneer P3-DX.

What's in each file?
File | Description
-----|-------------
SFLA | Contains the algorithm as a Python class
followsfla | A jupyter notebook to try the algorithm with a given list of obstacles. If needed, it includes the needed code to simulate the obtained path with Coppelia.
sensingsfla | This script is similar to followsfla, but instead of using a list of obstacles, it uses the readings from the sensors.
Obstacles.ttt | The scene used in coppeliasim. The the initial position of the robot and the goal can be changed.
pathcontrol | helper functions used to follow the path

## Some screenshots
![simulation](/Codigo/Salto_Rana/img/simulation.png)

### Paths obtained using followsfla
![Figure_5](/Codigo/Salto_Rana/img/Figure_5.png)
![Figure_7](/Codigo/Salto_Rana/img/Figure_7.png)

### Path obtained by using the robot sensors

![Figure_11](/Codigo/Salto_Rana/img/Figure_11.png)

## Some useful links

* [A survey on shuffled frog leaping algorithm](https://towardsdatascience.com/a-survey-on-shuffled-frog-leaping-algorithm-d309d0cf7503)

* [CoppeliaSim Remote API for Python](https://www.coppeliarobotics.com/helpFiles/en/remoteApiFunctionsPython.htm)
