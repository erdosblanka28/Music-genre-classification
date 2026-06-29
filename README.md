# Music-genre-classification
<br>

## Project Description
This project aims to develop a machine learning model capable of classifying audio tracks into distinct musical genres using the GTZAN dataset. We will use Digital Signal Processing to extract MFCC features from audio files and implement and compare several machine learning algorithms inclusing K-Nearest Neighbors (KNN), Support Vector Machine, Random Forrest and MLP Neural Network to classify songs into 10 different categories. The goal is to evaluate the model's accuracy in identifying musical patterns across genres like Rock, Jazz, Classical, etc.
<br>
<br>
## Feature extraction

### MFCC (Mel-Frequency Cepstral Coefficients) - librosa library
To extract useable data from our [GTZAN Dataset](http://web-wp.archive.org/web/20220120112420/http://marsyas.info/downloads/datasets.html), we use MFCC. This is important beacuse with this technology we can convert the data into numbers. <br>In this project we are interested in the genre of the music. The genre of music is determined by the timbre. Humans don't hear linearly, meaning that we can precieve small differences between low tones well, for example between 100 Hz and 200 Hz, however, between high tones such between as 1000 Hz and 1100 Hz we cannot preceive small difference well. This is why we should convert the frequencies onto Mel-scale, which mimics the human hearing & logarithmize them also as humans precieve volume logarithmically. We also need to amplify higher frequency sounds because the low tones tend to overwhelm the high tones. <br>For this project we are also going to cut the 30 sec long sound into ~25 ms long sounds because this way they count as staionary singal with which we will be able to work & of course, to convert it into frequencies we are going to use Fast Fourier Transform. After everything we use Discrete Cosine Transform for data compression by discarding noise & eliminating redundant information.<br>All this is included in the librosa library thus we dont have to program it manually.
```python
 amplitude, sampling_rate = librosa.load(new_path)
    mfcc_1 = librosa.feature.mfcc(y=amplitude, sr=sampling_rate, n_mfcc=20) #cuts the songs into ~25ms pieces, saving the 20 first coefficient
    mfcc_mean = np.mean(mfcc_1, axis=1) #taking the mean for each row (20 seperate mfcc value averaged over the whole song)
```
<br>From scientific sources we know that the most important information in human speech and music, the most distinguishable to our ears (accents, formants, timbres) is concentrated in the first 13-20 coefficients. Thus we chose it to be 20 to make it as sensitive as it can be while we also want to keep the system noise-free.

### Data converting & saving
After we extracted the data, we converted them to NumPy arrays. We compressed them and saved them into "music_features.npz" file so later it will be easier & computationally cheaper to use the data for different train models.
```python
np.savez_compressed("music_features.npz", features=x,labels=y)
```
<br>

## Train models

### 1. K-Nearest neighbours
![Alternatív szöveg](knn_k_value_test_13.png)
![Alternatív szöveg](knn_k_value_test_20.png)
![Alternatív szöveg](knn_k_value_test_40.png)
![Alternatív szöveg](knn_k_value_test_60.png)

### 2. Support Vector Machine

A Support Vector Machine model was implemented as a second supervised classification method. Since SVM is sensitive to the scale of the input features, the MFCC feature vectors were standardized using `StandardScaler` before training. An RBF kernel was used because the music genres are not expected to be linearly separable in the MFCC feature space.

The SVM model was tested using 13, 20, 40, and 60 MFCC coefficients. The best result was achieved using 40 MFCC features, with an accuracy of 67.00%.

| Number of MFCC features | SVM accuracy |
|---|---|
| 13 | 64.00% |
| 20 | 65.00% |
| 40 | 67.00% |
| 60 | 65.50% |

![SVM accuracy by MFCC features](svm_accuracy_by_features.png)

The confusion matrix shows that the model performs especially well on genres such as classical, blues, metal, jazz, and pop. The most difficult genres were disco, rock, country, and reggae, which were often confused with similar genres. This is expected because these genres can have overlapping timbre and rhythm characteristics when represented only by averaged MFCC features.

![SVM confusion matrix](svm_confusion_matrix_40.png)

### 3. Random Forest

A Random Forest classifier was implemented as a third supervised machine learning model. Random Forest uses multiple decision trees and combines their predictions to improve classification performance. Unlike SVM, it does not require feature scaling because decision trees split the data based on feature thresholds.

The model was tested using 13, 20, 40, and 60 MFCC coefficients. The best result was achieved using 40 MFCC features, with an accuracy of 66.00%.

| Number of MFCC features | Random Forest accuracy |
|---|---|
| 13 | 63.00% |
| 20 | 64.50% |
| 40 | 66.00% |
| 60 | 63.50% |

![Random Forest accuracy by MFCC features](random_forest_accuracy_by_features.png)

The confusion matrix shows that Random Forest performed well on classical, metal, pop, jazz, and hiphop. The most difficult genres were disco, country, and rock, which were often confused with other genres (similar to). This is expected because these genres can have similar timbre and rhythm characteristics when represented only by averaged MFCC features.

![Random Forest confusion matrix](random_forest_confusion_matrix_40.png)

The feature importance plot shows that the lower MFCC coefficients contributed the most to the classification. This is reasonable because the first MFCC coefficients contain the strongest information about the general spectral shape and timbre of the audio signal.

![Random Forest feature importance](random_forest_feature_importance_40.png)

### 4. MLP Neural Network

A simple neural network was implemented as an additional supervised classification model. The model is a Multi-Layer Perceptron (MLP), which uses the already extracted MFCC feature vectors as input. This is different from a CNN, because the model does not use spectrogram images, but averaged MFCC coefficients.

Since neural networks are sensitive to the scale of the input data, the MFCC features were standardized using `StandardScaler` before training. The genre labels were also encoded into numerical values using `LabelEncoder`.

The neural network was tested using 13, 20, 40, and 60 MFCC coefficients. The best result was achieved using 40 MFCC features, with an accuracy of 64.00%.

| Number of MFCC features | Neural Network accuracy |
| ----------------------- | ----------------------- |
| 13                      | 63.00%                  |
| 20                      | 60.50%                  |
| 40                      | 64.00%                  |
| 60                      | 58.50%                  |

![Neural Network accuracy by MFCC features](neural_network_accuracy_by_features.png)

The confusion matrix shows that the neural network performed well on classical, blues, pop, metal, and jazz. The most difficult genres were disco, rock, hiphop, country, and reggae, which were often confused with other genres. This is expected because these genres can have overlapping timbre and rhythm characteristics when represented only by averaged MFCC features.

![Neural Network confusion matrix](neural_network_confusion_matrix_40.png)

The training loss curve shows that the neural network successfully learned from the training data, since the loss decreased steadily during training. However, the final accuracy was lower than SVM and Random Forest, which suggests that this simple MLP model is not the best classifier for the current averaged MFCC feature representation.


![Neural Network training loss](neural_network_metrics_curve_40.png)

## Final Model Comparison

After implementing all models, we compared the performance of KNN, SVM, Random Forest, and an MLP Neural Network. All models were trained and tested using the same 80/20 train-test split, with stratification to preserve the genre distribution in both the training and test sets.

The models were tested using 13, 20, 40, and 60 MFCC coefficients. This allowed us to compare both the effect of the model type and the effect of the number of MFCC features.KNN used the fixed k = 27 baseline.



| Number of MFCC features |              KNN  |    SVM | Random Forest | Neural Network |
| ----------------------- | ------------------| -----: | ------------: | -------------: |
| 13                      |            55.50% | 64.00% |        63.00% |         63.00% |
| 20                      |            46.00% | 65.00% |        64.50% |         60.50% |
| 40                      |            57.00% | 67.00% |        66.00% |         64.00% |
| 60                      |            58.00% | 65.50% |        63.50% |         58.50% |



![Model accuracy comparison](comparison_all_models_by_features.png)

The best overall model was the Support Vector Machine using 40 MFCC features, achieving an accuracy of 67.00%. Random Forest performed very similarly, reaching 66.00% accuracy with the same number of MFCC features. The MLP Neural Network achieved 64.00%, while KNN achieved its best result using 60 MFCC features, with an accuracy of 58.00%.

![Best accuracy by model](comparison_best_accuracy_by_model.png)

The comparison shows that KNN is the weakest model. This is most likely because KNN makes decisions based directly on distances between feature vectors, while SVM with an RBF kernel can learn a nonlinear decision boundary between overlapping genre classes.

Since 40 MFCC features gave the best result for SVM, Random Forest, and the Neural Network, we also compared the models directly at this feature size.

![Model comparison using 40 MFCC features](comparison_40_mfcc_features.png)

The heatmap shows that the strongest model-feature combination was SVM with 40 MFCC features. Increasing the number of MFCC features to 60 did not improve the advanced models. This suggests that the additional MFCC coefficients may contain less useful information or introduce noise into the classification.

![Accuracy heatmap](comparison_accuracy_heatmap.png)

The SVM model performed especially well on classical, blues, metal, jazz, and pop. The most difficult genres were disco, country, rock, and reggae. These genres were often confused with other genres, which is expected because they can have overlapping timbre and rhythm characteristics when represented only by averaged MFCC features.

Overall, the best performing model was the Support Vector Machine, which achieved the highest accuracy of 67.00% using 40 MFCC features. Random Forest was very close with 66.00%, showing that it is also suitable for this classification task. The MLP Neural Network performed slightly worse with 64.00%, probably because the input data only contained averaged MFCC features instead of richer time-frequency information such as spectrograms. The KNN model only reached 58.00% accuracy, making it the weakest of the tested models. This suggests that for the current MFCC-based feature representation, SVM is the most effective classifier because its RBF kernel can handle nonlinear separation between overlapping music genres better than distance-based KNN.

## Future Improvements

There are several ways the project could be improved in the future. First, more audio features could be added besides MFCCs, such as chroma features, spectral contrast, zero-crossing rate, tempo, and rhythm-based features. These could help distinguish genres that have similar timbre but different rhythmic structure. Second, instead of averaging the MFCCs over the whole song, frame-level features could be used so that the model keeps more information about how the music changes over time. Third, a CNN could be implemented using spectrogram or mel-spectrogram images, which may perform better than the MLP model because it can learn patterns in the time-frequency domain. Finally, more hyperparameter tuning could be applied to all models, especially SVM and Random Forest, to check whether their accuracy can be improved further.

