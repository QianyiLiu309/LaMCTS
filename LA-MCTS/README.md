## Project repo for R244 "Exploring Parallel Bayesian Optimisation with LA-MCTS"

### Content
* Design and implement two parallelisation strategies in LA-MCTS for high-dimensional BO. 
* Integrate LA-MCTS with BoTorch and implement a novel gradient-based optimisation subroutine. 
* Evaluate the performance of parallel LA-MCTS and LA-MCTS (BoTorch) on synthetic functions and MuJoCo locomotion tasks.

### Project description

This project investigates the parallel-optimisation opportunities within LA-MCTS and used two approaches to parallelise LA-MCTS (root-parallelisation and leaf-parallelisation). With thorough evaluation on MuJoCo tasks, we demonstrate the performance improvement brought by both approaches and concludes that root-parallel LA-MCTS is more effective in this setting, due to its ability to improve both exploration and exploitation aspects of LA-MCTS. This project also makes an insightful study on integrating LA-MCTS with BoTorch. Despite not being able to fully achieve parallel tuning with BoTorch due to limited functionality, we manage to use BoTorch to implement a novel gradient-based optimisation routine, which further improves the sample-efficiency of LA-MCTS in high-dimensional black-box optimisation. We believe that this work makes novel contribution to the field, paving the way for further study on effective parallelisation of MCTS-based high-dimensional optimisation approaches. 