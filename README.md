# AutoScenario

## Overview
**AutoScenario** is an automated framework designed to improve the safety and reliability of **autonomous vehicle (AV) systems** through **realistic corner case generation**. Leveraging a **multimodal Large Language Model (LLM)**, it transforms **safety-critical real-world data** into structured scenario representations, enabling **generalization of key risk factors** for robust AV validation.

By integrating with **Simulation of Urban Mobility (SUMO) and CARLA**, AutoScenario automates **scenario generation, execution, and evaluation**, ensuring diverse and challenging test cases for AV systems. The framework supports **multimodal inputs(Text, Image and Video)**, making it an adaptable approach to **simulating rare and high-risk driving conditions**.

---

## Key Features
- 🚗 **Realistic Corner Case Generation**  
  Uses **LLMs** to synthesize **high-risk driving scenarios** from real-world data.  
- 📡 **Multimodal Input Processing**  
  Converts **Multimodal data or scene descriptions** into structured scenario representations.  
- 🏙 **Seamless Integration with SUMO & CARLA**  
  Automates the creation of AV test scenarios in standard simulators.  
- 📊 **Scenario Evaluation & Analysis**  
  Provides  **performance metrics** for generated scenarios.  

---

## Installation
### Prerequisites
- Python 3.8+
- **Recommended**: Use a virtual environment (Conda or virtualenv)

### **Step 1: Create & Activate Virtual Environment (Recommended)**
Using **Conda**:
```sh
conda create --name autoscenario python=3.8
conda activate autoscenario
```
Or using **virtualenv**:
```sh
python -m venv autoscenario
source autoscenario/bin/activate  # On macOS/Linux
autoscenario\Scripts\activate     # On Windows
```

### **Step 2: Install Dependencies**
Once the virtual environment is activated, install the required dependencies:
```sh
pip install -r requirements.txt
```
---

## **Usage**
### **Step 1: Set Up API Key and Configuration**
Before running simulations, configure your API key:

1. **Copy the example configuration file:**
   ```sh
   cp config.example .env
   ```

2. **Edit the `.env` file** with your actual API credentials:
   ```
   OPENAI_KEY=your-openai-api-key-here
   OPENAI_URL=https://api.openai.com/v1/chat/completions
   OPENAI_MODEL=gpt-4o
   OPENAI_MAX_TOKENS=2000
   OPENAI_TIMEOUT=30
   ```
3. Save the file in the project root directory.

---

### **Step 2: Run the Main Simulation**
You can run different **scenario generation modes** based on your input type(- Test samples can be found in the **data** folder):

#### **Command-Based Generation**
```sh
python experiments/auto_generate_all_command.py
```
#### **Long-Text-Based Generation**
```sh
python experiments/auto_generate_all_text.py
```
#### **Image-Based Generation**
```sh
python experiments/auto_generate_all_vlm.py
```
#### **Video-Based Generation**
```sh
python experiments/auto_generate_all_video.py
```


### Run Scenario Evaluations
```sh
python evaluation/general_evaluator.py
```



---

## Project Structure
```
/AutoScenario
│── agents/           # Core source code for AV agents
│── experiments/      # Experiment scripts for scenario testing
│── tools/            # Utility and evaluation scripts
│── data/             # Multimodal input datasets
│── requirements.txt  # Python dependencies
│── .gitignore        # Git ignore rules
│── LICENSE           # Open-source license
│── README.md         # Project documentation
```

---

## Contributing
We welcome contributions! 🚀  
To contribute, please follow these steps:

1. **Fork the repository** 📌  
2. **Create a new branch** (`feature-branch`) 🔀  
3. **Commit your changes** (`git commit -m "Add feature X"`) ✅  
4. **Push to your branch** (`git push origin feature-branch`) ⏫  
5. **Create a Pull Request (PR)** 📝  

---

## Citation
If you use **AutoScenario** in your research or projects, please cite us:  
📄 **Qiujing Lu, Meng Ma, Ximiao Dai, Xuanhan Wang, and Shuo Feng.**  
*"Realistic Corner Case Generation for Autonomous Vehicles with Multimodal Large Language Model."*  
**arXiv preprint arXiv:2412.00243 (2024).**  

```bibtex
@article{lu2024realistic,
  title={Realistic Corner Case Generation for Autonomous Vehicles with Multimodal Large Language Model},
  author={Lu, Qiujing and Ma, Meng and Dai, Ximiao and Wang, Xuanhan and Feng, Shuo},
  journal={arXiv preprint arXiv:2412.00243},
  year={2024}
}
```

Your citation **supports our work** and helps improve **future development** of AutoScenario! 🚀

---

## License
This project is **open-source** and licensed under the **MIT License**. See the `LICENSE` file for details.

---

## Contact & Support
💬 **For issues or collaborations, please:**  
- **Open an issue** on GitHub  
- **Reach out to the maintainers**  

🚀 **Let’s build safer AV systems together!** 🚗💨  
