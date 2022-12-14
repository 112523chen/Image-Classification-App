# Image-Classification-App

![Project Demo](demo.png)

## Project Description
Created a web app that conatins a machine learning model(s) that predicts what inside an image based on a few presets 

## Python Libaries Used
[Pandas](https://pandas.pydata.org), [Numpy](https://numpy.org), [Sklearn](https://scikit-learn.org/stable/), and [Streamlit](https://streamlit.io)

## Data Source
[Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
> A dataset from Kaggle that contains over 17,000 images of buildings, forests, glaciers, mountains, bodies of water, and streets

## File Directory
- ### Archive Folder 
  > Location of the images
- ### app.py
  > Location of the code for the web app and machine learing model(s)
- ### requirements.txt
  > Text file with details on enviroment requirements for the web app

## Future Updates:
- Displaying statstics about the current model used
- User interface change with a sidebar containing the following
  - Toggle to use different machine learning model
  - Toggle to change the train test split for the model
  - Toggle to add additional following

## How to Run Locally
- Download this repo and all packages mentioned in `requirements.txt`
- Go to terminal/command line and run the following command:
  `streamlit run app.py`
