# SeeWood
SeeWood is a forest species recognition system

# About
I use a pre-trained convolutional neural network on ImageNet and fine-tune the parameters on my own target dataset. The model achieves 96.03% of accuracy. I split every source image into 16 sub-images, taking advantage of isotropy of texture, then extend the training data efficiently for preventing the information completely. For the prediction, the test image is divided into 16 non-overlapping sub-images too, and each one is predicted individually. I collect the votes to produce the finial result using the Sum rule, which increase the accuracy 4.17%. This strategy is same as the one proposed by Filho et al. [1]. 

# Database
The database of macroscopic I use here contains 41 species of the Brazilian flora which were cataloged by the Laboratory of Wood Anatomy at the Federal University of Parana (UFPR)  in Curitiba, Brazil, and it is available upon request for research purpose [2]. 

# Acknowledgement
I appreciate Prof. Oliveira and the term who made a great effort to this area.

# Reference
[1] P. L. P. Filho, L. S. Oliveira, S. Nisgoski, and A. S. Britto, “Forest species recognition using macroscopic images,” Machine Vision and Applications, Jan. 2014
[2] http://web.inf.ufpr.br/vri/image-and-videos-databases/forest-species-database-macroscopic
