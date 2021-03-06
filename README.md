# Probabilistic ICP 

Probabilistic variant of the iterative closest point algorithm.
The point to plane matching is carried out using the Mahalanobis distance between the distributions from which the points in the PCLs have been drawn.

## Dependencies (Ubuntu 16.04)
* PCL  http://pointclouds.org/
* EIGEN http://eigen.tuxfamily.org/


## Building

Clone this repository and create a `build` folder under the root, then execute
```
cd build
cmake ..
make -j4
```

### Available apps
Avalible under the `bin` folder. Run
```
./test_picp ../meshes/monkey.ply
```
And hit space until the algorithm reaches convergence and 'q' to exit.
