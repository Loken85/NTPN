# NTPN: Neural Trajectory Point Net

### By Adrian Lindsay
### University of British Columbia

This is the home of the Neural Trajectory Point Net (NTPN), a spatial transformer based neural network approach to extracting and analysing the geometry of neural representations. 

The details and background of the NTPN can be found in my doctoral dissertation: Understanding the Geometry of Emotion [Lindsay, 2024: UBC] (https://doi.org/10.14288/1.0445225)


In it's current form, this repository is not stand alone, but is simply a collection of the functions and scripts used to perform the analysis described in the paper linked above.
Further development is in progress and will be available as a pre-print in future. 

## Basic Architecture of the NTPN
![ntpn_flowchart_captioned](https://github.com/user-attachments/assets/13b0d38f-4501-4694-91f1-4fb4db3ccad6)

The Neural Trajectory Point Net: Step-by-step 
1. Binned neuron spikes from electrophysiology recordings are embedded as fixed length temporal trajectories in neural space (each dimension of this space corresponds to the firing rate of a single neuron)
2. These neural trajectories are input to the first connected layer of the network.
3. The first of two spatial transformers. Transformers weigh each element of an input according to what is important or common. Here it creates a transformation matrix to appropriately weight spike count trajectories based on commonalities across all trajectories. This serves to align the inputs based on learned patterns of firing. The transform is agnostic of the identity of contributing neurons and the order of the trajectories, and is robust to the scaling and relative position of the patterns of interest. 
4. A multi-layer perceptron (MLP). MLPs are trained to find common patterns or features. In this case, the MLP is trained to recognize common trajectory shapes (features) from the aligned inputs it receives from (3). It maps or embeds the inputs from (3) into this new feature space. This network is shared, meaning each input undergoes an identical and independent embedding.
5. The second spatial transformer performs another transformation, this time on the embedded features from (4). This second transformation serves to align learned structure in the feature space, as opposed to the first transformer which aligned in input space. These transforms in conjunction provide invariance to rotations, translation, and scaling in the inputs/features. 
6. The second MLP. Features are again embedded into a higher dimension space by a second shared MLP. This MLP works as a pattern detector for the aligned feature space received from (5). Just as with the first, each feature is embedded by the same network, ensuring order invariance in the embedding. 
7. The high dimensional feature space is reduced to a set of global features via a max pooling operation, yielding an aggregate signal of the entire input trajectory. Applying only symmetric operators ensures that the resulting set of global features is invariant to input order and identity.
8. A three-layer fully-connected MLP classifier predicts a set of class probabilities (in our case it could be the 3 emotional contexts or a set of behaviors) from the global features. 
9. These class probabilities form the output of the network, predicting behaviours and emotional contexts from a set of neural trajectories. The structure of these trajectories is reflective of the dynamics of the neural system. The NTPN is invariant to perturbations of these trajectories by rotation, translation, and scaling, is robust to noise, and critically is invariant to the order (and identity) of the neurons whose firing makes up the input neural trajectories. Together, these allow it to learn to predict behaviour and context across both experimental sessions and subjects from neural data without alignment or sorting. 

A. A stylized example of how neural trajectories are transformed by the point net into a set of global features. Illustrated is the passage of a set of 1D trajectories through each step of the network. First, the inputs are aligned for local structure via a spatial transformer. These aligned inputs are then embedded into feature space by a set of shared filters. The embedded features are aligned for structure in feature space by a second spatial transformer. A second feature embedding generates a set of meta-features, or features of features. This final feature space is then collapsed to a set of global features by pooling the maximal responses. 

B. Relevant structure of the input neural trajectories and their underlying neural manifold can be extracted from the network. Critical sets: the subset of each trajectory that was informative to the network in assigning it an output class. Upper-bounds: the bounds of each class as set by the invariances of the network; samples of a given class lie between the minimal ‘critical set’ and the ‘upper-bound’. Topology: the extracted features are examples of high-dimensional geometry, shapes, for each of the associated neural manifolds. These shapes can by analysed further for their topology and other geometric properties. 




## Applying Topological Data Analysis (TDA) to the output of the NTPN

### Critical sets and bound shapes. 
![critical_sets_and_bound_shapes](https://github.com/user-attachments/assets/8c187357-b9c7-46c1-82eb-1e0845e80d35)

A) Schematic of extracting a critical set for a given example input trajectory. For each input, the trained NTPN is queried for the points in the trajectory that are maximally responsive on each feature (points which influence the global feature vector). Those critical points form a subset of the original trajectory, a critical set. 

B) Critical sets are aligned with each other via a shared UMAP projection followed by CCA. Plotted are three examples groups of aligned critical sets. C) Bound shapes are generated from a set of aligned critical sets, all belonging to the same class. This set of sets defines the space of the underlying manifold that is visited across the population of trajectories and is relevant to defining the class (as it is a combined set of critical points). This set of sets is spatially sampled to produce a bound for the manifold, which we visualise as a voxelized partial surface. 

### Interpreting Neural TDA
![interpreting_neural_tda_schematicv2](https://github.com/user-attachments/assets/132deb4a-706f-4be8-93e0-a30a5aea330d)

### Comparing Neural Manifolds with Mapper
![comparing_with_mapper_bw](https://github.com/user-attachments/assets/bd57cfa1-77ca-4f0d-a1fa-0f456495cb53)

Differences in the geometry of neural representations: Mapper. 
2-D views of 3-D graph/map plots from the ‘Mapper’ algorithm. Bound shapes of emotional context representations and select behaviour representations were processed as point clouds by Mapper to produce graph representations of topology. 
Point clouds were filtered using a 3-D UMAP projection, and intervals were built with a cubical cover. Clustering was done using a density-based clustering method. 
The resulting structure based topology was graphed in 3-D using the same 3-D UMAP projection. Scale is relative for each Mapper projection. 
A-C) Mapper visualisations of emotional context representations. Structure within the graphs is representative of the geometry of the bound shapes for each of the contexts. 
D-F) Mapper visualisations of behaviour representations for 3 select behaviours. Again, structure in the graphs is indicative of the unique geometric structure of the neural representation of that behaviour. There are marked differences between the structure of these neural representations, in particular between emotional contexts as a group and behaviours.   

Images and captions from [Lindsay, 2024: UBC] (https://doi.org/10.14288/1.0445225)



