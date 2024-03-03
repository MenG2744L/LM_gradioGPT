from gradio_client import Client

client = Client("http://127.0.0.1:7860/")
result = client.predict(
    "Howdy!",  # str in 'Text' Textbox component
    0.2,  # int | float (numeric value between 0.1 and 1) in 'SDP/DP混合比' Slider component
    0.5,  # int | float (numeric value between 0.1 and 1) in 'noise' Slider component
    0.8,  # int | float (numeric value between 0.1 and 1) in 'noisew' Slider component
    1.0,  # int | float (numeric value between 0.1 and 2) in 'length' Slider component
    "LiuMeng",  # str (Option from: ['LiuMeng']) in 'character' Dropdown component
    fn_index=0
)
print(result)
