## Pre processing and feature selection

It is necessary to find a balance between reducing input data - which can greatly improve processing time - while retaining data that provides the most relevant information. Some questions:

- are 3 channels necessary? (grayscaling)
- can we use a lower resolution? (downscaling)
- can we only use a section of a picture? (maintain resolution, but decreases data -> take only stripes/corners?)
- do we really need all data?
    - can we make an intensity distribution by taking quantiles?
        - remove the 0th and 99th quantile to exclude outliers
        - can we use more quantiles??
    - can we flatten the data to a single dimension? (loses spatial information)

## Algorithm choice

1. Conventional ML:
    - should receive pre processed data
    - uses flattened data -> have to do intelligent preprocessing and feature selection
    - can we use OpenCV's edge detection to distinguish between categories?
    - can we use OpenCV's star detection by taking bounding boxes and flattening?
    - do it manually? Sobel operator, canny edge detection, laplacian variance
    - multiple algorithms can be used and parallelized! some may be better than others for different classes
        - ensemble learning and soft voting to given classes
        - use a router/meta-model to find features and send it to appropriate model
    - `SGDClassifer`: uses SGD to optimize linear models
    
2. Deep learning:
    - *can* receive raw data: model is "smarter" (has inductive bias)
    - CNNs are natural choice for images
    - consider the model's structure (hyperparameters):
        - number of layers
        - kernel size (size of the filter to extract features from images)
        - number of feature maps: amount of output filters (channels) in a convolutional layer.
        Each filter learns to detect different features. The more filters, more complex features and more compute.
        - stride: number of pixels by which the filter moves
        - pooling: takes min/average/max at each window to downsampling/downscaling.
        Helps with overfitting and memory usage
    - take in consideration multiply accumulates (MACs) and edge exclusions in CNN layers


## Training and testing

- balance classes to avoid bias
- some labels may be ambiguous -> check the model's confusion matrix
- **reduce training test size (with balanced labels!) to evaluate different models faster**

## Metrics

### Precision
indicates how many predicted positive values are, indeed, positive

$$
\frac{TP}{TP+FP}
$$


### Recall
Indicates how many actual positive values were correctly predicted as positive

$$
\frac{TP}{TP+FN}
$$

## Output of an artificial neuron
$f\left(\sum_i W_i X_i + b\right)$

$f$ provides non-linearity while the sum is a **linear combination** of weights, values, and bias.

Over epochs, the data builds the weights -> picking an architecture that maximizes this is key