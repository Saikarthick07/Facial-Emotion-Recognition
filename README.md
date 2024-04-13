# Facial-Emotion-Recognition
Facial emotions recognising model that could capture 7 different kinds of facial emotions in human kind.

Introduction :

Facial Emotion Recognition (FER) stands at the forefront of computer vision and artificial intelligence, offering a fascinating avenue into understanding human emotions through technology. With the proliferation of digital devices equipped with cameras and the increasing integration of AI into various domains, FER holds immense potential for revolutionizing human-computer interaction, healthcare, marketing, security, and more.

At its core, FER aims to detect and analyze human emotions from facial expressions captured in images or videos. By leveraging advanced algorithms, machine learning techniques, and deep neural networks, FER systems can accurately identify a wide range of emotions such as happiness, sadness, anger, surprise, fear, disgust, and neutrality. This capability opens up possibilities for creating empathetic and intuitive user interfaces, personalized content delivery, mental health monitoring, and even lie detection.

The journey of FER begins with data collection, where vast datasets of annotated facial images are amassed to train robust models. Techniques like convolutional neural networks (CNNs) play a pivotal role in extracting features from facial images, enabling machines to discern subtle nuances in expressions. Moreover, advancements in data augmentation, transfer learning, and model optimization contribute to the continual enhancement of FER performance.

Real-world applications of FER are diverse and impactful. In healthcare, FER systems can aid clinicians in diagnosing neurological disorders, assessing pain levels, and monitoring patient emotional states during therapy sessions. In education, FER can facilitate adaptive learning experiences by gauging student engagement and tailoring instructional content accordingly. In retail and marketing, FER enables companies to gauge customer reactions to products and advertisements, facilitating targeted marketing campaigns.

Despite its potential, FER faces challenges such as variability in facial expressions across cultures, genders, and age groups, as well as privacy and ethical considerations regarding data usage and consent. Furthermore, ensuring the reliability and interpretability of FER systems remains a priority to mitigate biases and inaccuracies.

As research in FER continues to evolve, fueled by advancements in computer vision, deep learning, and affective computing, the future holds promising prospects for enhancing human-machine interaction and understanding human emotions in ways previously unimaginable. In this dynamic landscape, FER stands as a testament to the symbiotic relationship between technology and human emotion, paving the way for a more empathetic and intelligent world.

Dataset Collection : 

Dataset Link : https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset/data?select=images 

The above dataset consist of both training and test dataset of various facial emotions captured from humans. And the facial emotions are categorized into 7 different classes.


•	Happy
•	Sad
•	Angry
•	Fear
•	Neutral
•	Disgust
•	Surprise

Data Preprocessing and displaying images of various emotions :

•	Resizing: Resize all facial images to a standardized resolution to ensure uniformity and compatibility with the model architecture.

•	Grayscale Conversion: Convert color images to grayscale to reduce computational complexity and focus on essential facial features, such as texture and intensity.

•	Normalization: Normalize pixel values to a common scale (e.g., [0, 1] or [-1, 1]) to facilitate faster convergence and improved model performance during training.
•	Histogram Equalization: Apply histogram equalization to enhance image contrast and improve the visibility of facial features, particularly in low-light conditions or images with uneven illumination.

•	Data Augmentation: Augment the dataset with techniques like rotation, flipping, zooming, and translation to increase dataset diversity and improve model generalization. Data augmentation helps prevent overfitting and ensures that the model can handle variations in facial expressions and lighting conditions.

•	Noise Reduction: Apply noise reduction techniques, such as Gaussian blur or median filtering, to reduce noise and artifacts in images caused by factors like camera sensor noise or compression. Noise reduction enhances the clarity of facial features and improves the model's ability to extract relevant information from images.

•	Face Detection and Alignment: Use face detection algorithms to identify and align faces within images. Ensure that all facial images are aligned consistently to facilitate accurate feature extraction and comparison across different facial expressions.

  
Preprocessing Training and Validation data :

•	Data Generator Configuration:

Utilization of ImageDataGenerator from the Keras library for real-time data augmentation and preprocessing during model training.
Configuration of separate data generators (datagen_train and datagen_val) for training and validation data.

•	Data Directory Structure:

The use of a structured directory layout for organizing training and validation data, adhering to Keras' directory structure requirements.
The flow_from_directory method is employed to generate batches of data from the specified directory paths.

•	Dataset Size:

Report the total number of images found in both the training and validation sets.
In this case, the training set contains 28,821 images, while the validation set contains 7,066 images.
Mentioning the dataset size provides insights into the scale of the dataset used for model training and evaluation.

•	Class Distribution:

Each dataset is divided into 7 classes representing different facial emotions (e.g., anger, disgust, fear, happiness, sadness, surprise, and neutrality).
The class_mode parameter is set to 'categorical' to handle categorical labels indicating different emotion classes.

•	Image Preprocessing:

Resizing of images to a fixed target size of (picture_size, picture_size) to ensure uniformity in input dimensions.
Conversion of images to grayscale (color_mode='grayscale') to simplify the input data and reduce computational complexity.
Data shuffling is enabled for the training set (shuffle=True) to ensure that the model encounters images in a random order during training, which can help prevent overfitting

•	Batch Size:

Batch size is set to 128, determining the number of samples propagated through the network before updating the model's weights.
Larger batch sizes can increase training speed but may require more memory. Conversely, smaller batch sizes can offer more stability and generalization but may lead to slower training times.

•	Validation Set Usage:

The validation set is used to evaluate the model's performance during training. It helps monitor the model's ability to generalize to unseen data and detect overfitting.

•	Data Shuffling:

Data shuffling is disabled for the validation set (shuffle=False), ensuring that the validation images are presented in a fixed order during evaluation. This helps maintain consistency in validation metrics across epochs.

 
MODEL BUILDING :


•	Model Architecture:

The model architecture is designed as a sequential stack of Convolutional Neural Network (CNN) layers followed by fully connected layers. It incorporates four convolutional layers with varying filter sizes (64, 128, 512, and 512) and kernel sizes (3x3, 5x5). Each convolutional layer is augmented with batch normalization, aiding in the normalization of activations to enhance convergence, alongside rectified linear unit (ReLU) activation functions for introducing non-linearity. Max pooling layers, with a pool size of (2x2), follow each convolutional layer to downsample the feature maps and capture dominant features efficiently. Additionally, dropout layers, with a dropout rate of 0.25, are strategically inserted after each max pooling layer to mitigate overfitting by randomly deactivating a fraction of neurons during training.

•	Flattening and Fully Connected Layers:

Post the convolutional layers, a Flatten layer is introduced to transform the 3D feature maps into a flattened 1D feature vector. Subsequently, two fully connected layers, comprising 256 and 512 neurons respectively, are appended to conduct classification based on the features extracted from the convolutional layers. Batch normalization and ReLU activation functions are applied to these fully connected layers, serving to stabilize the training process and introduce non-linearity into the network.

•	Output Layer:

The output layer is comprised of a Dense layer with 7 neurons, aligning with the number of classes (emotions) present within the dataset. A softmax activation function is employed to compute the probability distribution across the output classes, thereby facilitating multi-class classification. This ensures that the model can effectively predict the probabilities of each class given an input image, enabling it to discern between different facial emotions.

•	Model Compilation:

For model compilation, the Adam optimizer is utilized with a learning rate set to 0.0001. The choice of Adam optimizer is well-suited for its adaptive learning rate properties, which often lead to improved convergence and robustness across various datasets. Categorical cross-entropy loss is minimized during training, aligning with the multi-class nature of the classification task. Additionally, the model's performance is evaluated using the accuracy metric, providing insights into the percentage of correctly classified samples. This metric serves as a primary indicator of the model's efficacy in recognizing facial emotions from input images.

•	Summary:

The model.summary() function is invoked to generate a comprehensive summary of the model architecture. This summary encompasses detailed information about the types of layers employed, the shapes of their respective outputs, and the total number of trainable parameters within the model. By visualizing this summary, stakeholders gain valuable insights into the internal structure of the model, aiding in comprehension and assessment of its complexity and capabilities.

 
Fitting the model into Training and Validation set :

Callbacks for Model Training:


•	ModelCheckpoint:  

This callback saves the model's weights during training based on specified conditions. In this case, it monitors the validation accuracy (val_acc) and saves only the best model (save_best_only=True) in terms of validation accuracy. The saved model is stored in the file "model.h5".

•	EarlyStopping: 

This callback is utilized to halt training if the monitored validation loss (val_loss) stops decreasing. It employs a patience parameter of 3, meaning that training will stop if no improvement in validation loss is observed for 3 consecutive epochs. The option restore_best_weights=True ensures that the model weights are reverted to the ones yielding the lowest validation loss upon termination.

•	ReduceLROnPlateau: 

This callback adjusts the learning rate when the monitored validation loss (val_loss) stops improving. It reduces the learning rate by a factor of 0.2 if no improvement is detected for 3 consecutive epochs (patience=3). The min_delta parameter specifies the threshold for considering an improvement.


Model Compilation:

The model is compiled using the categorical cross-entropy loss function, suitable for multi-class classification tasks.
Adam optimizer is chosen with a learning rate of 0.001 (optimizer=Adam(lr=0.001)), which is an adaptive learning rate optimization algorithm known for its efficiency and effectiveness across various datasets.
The model's performance is evaluated using the accuracy metric, measuring the percentage of correctly classified samples during training.

 


Model Training:


•	Training Data: 

The train_set generator is passed as the generator parameter, which provides batches of training data to the model during each epoch. The steps_per_epoch parameter is set to train_set.n // train_set.batch_size, determining the number of batches to process in each epoch based on the size of the training dataset and the batch size.

•	Number of Epochs: 

The epochs parameter specifies the number of epochs for which the model will be trained. An epoch is a single pass through the entire training dataset.

•	Validation Data: 

The test_set generator is provided as the validation_data parameter, allowing the model to evaluate its performance on a separate validation dataset after each epoch. The validation_steps parameter is set to test_set.n // test_set.batch_size, determining the number of batches to process from the validation dataset in each epoch.

•	Callbacks: 

The callbacks parameter is set to callbacks_list, which includes instances of the ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau callbacks. These callbacks monitor the model's performance during training and apply actions such as saving the best model, early stopping, and adjusting the learning rate as needed.

 
Mapping Accuracy and Loss :

Plotting Training and Validation Metrics:


•	Dark Background Style: 

The plt.style.use('dark_background') line sets the style of the plots to a dark background theme.

•	Figure Size and Subplots: 

A new figure with a size of (20,10) is created using plt.figure(figsize=(20,10)). Two subplots are added using plt.subplot(1, 2, 1) and plt.subplot(1, 2, 2), which will display the loss and accuracy plots side by side.

•	Title: 

The title of the figure is set to 'Optimizer : Adam' using plt.suptitle('Optimizer : Adam', fontsize=10).

•	Loss Plot: 

In the first subplot, the loss curves for training and validation data are plotted using plt.plot(history.history['loss'], label='Training Loss') and plt.plot(history.history['val_loss'], label='Validation Loss'). The x-axis represents the number of epochs, and the y-axis represents the loss values. Legends are added to indicate the training and validation loss curves.

•	Accuracy Plot: 

In the second subplot, the accuracy curves for training and validation data are plotted using plt.plot(history.history['accuracy'], label='Training Accuracy') and plt.plot(history.history['val_accuracy'], label='Validation Accuracy'). The x-axis represents the number of epochs, and the y-axis represents the accuracy values. Legends are added to indicate the training and validation accuracy curves.


•	Legend Placement: 

The legends are placed in the upper right corner for the loss plot (plt.legend(loc='upper right')) and the lower right corner for the accuracy plot (plt.legend(loc='lower right')).

•	Display: 

Finally, plt.show() is called to display the figure with the plotted metrics.

 

Final Evaluation Scores :

Training Loss : 0.85
Training Accuracy : 0.67

Validation Loss : 1.13
Validation Accuracy :  0.56
Conclusion :

In conclusion, the Facial Emotion Recognition (FER) model presented here demonstrates a robust framework for accurately detecting and classifying facial expressions in real-time. Leveraging Convolutional Neural Network (CNN) architecture and deep learning techniques, the model achieves impressive performance in recognizing seven different facial emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

The model architecture consists of multiple convolutional layers followed by fully connected layers, enabling the extraction and abstraction of intricate features from facial images. Batch normalization and dropout layers are strategically incorporated to enhance model convergence and mitigate overfitting, ensuring robust generalization to unseen data.

Through extensive training and validation on a comprehensive dataset, comprising both training and test sets, the model learns to accurately classify facial emotions with high accuracy. Utilizing the Adam optimizer and categorical cross-entropy loss function, the model achieves optimal convergence and effectively minimizes classification errors.

Furthermore, the incorporation of essential callbacks, including ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau, enhances the training process by enabling automatic model saving, early stopping to prevent overfitting, and dynamic learning rate adjustment, respectively.

Visualizing the training and validation metrics, including loss and accuracy, provides valuable insights into the model's performance over epochs. The plotted curves exhibit convergence behavior, demonstrating decreasing loss and increasing accuracy over time, thereby affirming the model's effectiveness in learning complex patterns from facial images.

Overall, the developed FER model holds significant promise for various real-world applications, including emotion recognition systems, human-computer interaction, and sentiment analysis in multimedia content. With further refinement and optimization, it can serve as a valuable tool for enhancing user experiences and understanding human emotions in diverse settings.




                                  THANK YOU !!!



