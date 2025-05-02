# Welcome to the Team 7 repository

This repository is made available to us to share resources and collaborate.  You will find some folders with information on the pilot chatbot previously created, as well as foundation information shared by N3xtcoder about the challenge.

### **Key files and references:**

- **Notion project notes:** https://www.notion.so/AI-Parenting-Assistant-Chatbot-Project-1d1f99943bee80cbb85bf497aa42996a

- **Deployed Web App (Azure):** https://myaibotappservice.azurewebsites.net/index.html

- **Project Repository (GitHub):** https://github.com/jeannineshiu/AiBot
- **AI 4 Impact Elternleben Repository (GitHub):** https://github.com/n3xtcoder/ai4impact-elternleben


### **Other files:**
- **Chatbot User Journey:** https://docs.google.com/drawings/d/1syz1xuTLoigGUUkENb1PSKR17G-uQ_nnRNlvSUhbAD0/edit?usp=drive_link
This is an in-progress document for us to update and refer to, as needed.  It should provide reference to the journey the user would take once entering the ChatBot. The file is created through Google Drawings and is open to edit by all who access the link.
- **Chatbot Timeframe:** https://docs.google.com/drawings/d/1Uql-fBloV_CWmxlomrYozs3d0hAJm9ReI_8hU6C0ddE/edit?usp=sharing
This is an overall timeframe with key milestones, open for edits and updating. Snapshot of image as of 22 April follows:

![Elternleben_ChatBot_Timeframe](https://github.com/user-attachments/assets/cdf16a57-e05b-4387-b99f-c8297d20ab54)

### ChatBot_Elternleben

This folder contains all the files and subfolders needed to run **ChatBot_Elternleben**, a chatbot built with Streamlit. It is intended for local testing while we continue development of a cloud-hosted version using Azure. If you have any questions, contact Edicta.

#### How to run the chatbot locally

1. **Download the full `ChatBot_Elternleben` folder** to your local machine.

2. **Create a free [Hugging Face](https://huggingface.co/) account** to obtain an access token:
   - Go to your profile → **Settings**
   - In the left sidebar, click **Access Tokens**
   - Click **New token**, give it a name, set the role to **read**, and generate it
   - **Copy the token immediately** — it will not be shown again

3. **Insert your Hugging Face token** into the script:
   - Open `ChatBot_Elternleben.py`
   - Paste the token on **line 110**

4. **Run the script from your terminal**:

   ```bash
   streamlit run [path-to-folder]/ChatBot_Elternleben/ChatBot_Elternleben.py
