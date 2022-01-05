#A Light in the Dark: Deep Learning Practices for Industrial Computer Vision

## Abstract
This is the repository for our Contribution to the WI2022 in Nuremberg.

In recent years, large pre-trained deep neural networks (DNNs) have revolutionized the field of computer vision (CV).
Although these DNNs have been shown to be very well suited for general image recognition tasks, application in industry is often precluded for three reasons: 
1) large pre-trained DNNs are built on hundreds of millions of parameters, making deployment on many devices impossible, 
2) the underlying dataset for pre-training consists of general objects, while industrial cases often consist of very specific objects, such as structures on solar wafers, 
3) potentially biased pre-trained DNNs raise legal issues for companies. 
   
As a remedy, we study neural networks for CV that we train from scratch. 
For this purpose, we use a real-world case from a solar wafer manufacturer. 
We find that our neural networks achieve similar performances as pre-trained DNNs, even though they consist of far fewer parameters and do not rely on third-party datasets. 

## Structure of this repository
```
+-- ImageClassification            | Runner Notebook + Scripts for experiments
+-- ReadMe.md			   | ReadMe
+-- Results.xlsx                   | Results that were reported in the paper
+-- RunResults                     | Detailed logging of our experiments results that were reported in the paper (IDs correspond to old IDs in the .xlsx file due to procedure)
```
