# Getting Started  
    Follow these steps to set up and run the API on your local machine. 

## Download Models 
    Download the models from [this DRIVE folder](https://drive.google.com/drive/folders/16Q4_GUVHssoPhvbNm5u_tANH0lKr5i0F?usp=drive_link), extract them, and place them in the project directory.
    

## Setup

    ### Create a virtual environment 
        Run the following command to create a virtual environment:
        `python -m venv .venv`

    ### Enter the virtual environment
        **For Windows(CMD/PowerShell) :** 
        `.\.venv\Scripts\activate`

        **For macOS/Linux :**
        `source .venv/bin/activate`

    ### Install requirements  
        Once inside the virtual environment, install the required dependencies:
        `pip install -r requirements.txt`
    
    ### Running the API
        To start the API, use the following command: 
        `python api.py` 
    
    ### Testing the API
        Use tools like Postman or cURL to test the API endpoints. Example (Windows Powershell):
        ` Invoke-RestMethod -Uri "http://127.0.0.1:5000/sentiment" -Method Post -Headers @{ "Content-Type"="application/json" } -Body '{"text": "This is amazing!"}`