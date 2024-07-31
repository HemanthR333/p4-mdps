 
import streamlit as st  
import tensorflow as tf  
  
@st.cache  
def load_model():  
    interpreter = tf.lite.Interpreter('detection.lite')  
    interpreter.allocate_tensors()  
    return interpreter  
 
def predict(input_image):  
    interpreter = load_model()  
    input_details = interpreter.get_input_details()  
    output_details = interpreter.get_output_details()  
    interpreter.set_tensor(input_details['index'], input_image)  
    interpreter.invoke()  
    output_data = interpreter.get_tensor(output_details['index'])  
    return output_data  
 
st.title('Object Detection with TensorFlow Lite')  
  
# Create a file uploader for the video capture device  
video_capture_device_id = st.file_uploader('Select a video capture device:', type=['mp4', 'avi'])  
  
if video_capture_device_id is not None:  
    # Create a video capture object  
    cap = cv2.VideoCapture(video_capture_device_id)  
  
    # Create a button to start the classification  
    if st.button('Start Classification'):  
        while True:  
            ret, img = cap.read()  
            if not ret:  
                break  
  
            # Preprocess the image  
            img = cv2.resize(img, (224, 224))  # assuming the model expects 224x224 input  
            img = img / 255.0  # normalize the image  
  
            # Run the prediction  
            output = predict(img)  
  
            # Process the output  
            labels = ["cadbury_DM","indomie_goreng","kitkat","kitkat_gold","mentos","milo_nuggets","pocky_chocolate","toblerone"]  # define the labels for the objects  
            prices = {"cadbury_DM": 1.1, "indomie_goreng": 0.4, "kitkat": 0.6, "kitkat_gold": 0.8, "mentos": 0.7, "milo_nuggets": 1.0, "pocky_chocolate": 1.2, "toblerone": 2.0}  
            total = 0  
  
            for bb in output:  
                label = bb['label']  
                score = bb['value']  
                x, y, w, h = bb['x'], bb['y'], bb['width'], bb['height']  
                print(f'\t{label} ({score:.2f}): x={x} y={y} w={w} h={h}')  
                total += prices[label]  
  
            # Display the output  
            st.write(f'Result: {len(output)} bounding boxes')  
            st.write(f'Total: ${total:.2f}')  
  
            # Display the image with bounding boxes  
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  
            for bb in output:  
                x, y, w, h = bb['x'], bb['y'], bb['width'], bb['height']  
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)  
            st.image(img, channels='BGR')  
  
            # Exit on key press  
            if cv2.waitKey(1) == ord('q'):  
                break  


