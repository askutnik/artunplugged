# Installation Instructions

1. Ensure you have Python installed on your system. You can download it from [python.org](https://www.python.org/).

2. Install `pip`, the Python package manager, if it is not already installed. You can check by running:
    ```
    pip --version
    ```
To install, run:
    ```
    py -m ensurepip --upgrade
    ```


3. Install the required packages listed in the requirements file:
    ```
    pip install -r requirements.txt
    ```


4. Run the programm by navigating to the src folder and running the Streamlit dashboard:
    ```
    cd src
    streamlit run streamlit_dashboard.py
    ```