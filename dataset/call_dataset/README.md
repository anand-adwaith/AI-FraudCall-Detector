# Llmfarminf Project

## Overview
The Llmfarminf project is designed to generate synthetic call transcripts using OpenAI's API. The primary goal is to create a large dataset of approximately 50,000 entries for research and development purposes.

## Project Structure
```
llmfarminf-project
├── src
│   ├── llmfarminf.py       # Contains the Llmfarminf class for API interaction and data generation
│   ├── main.py              # Entry point for the application to generate call transcripts
│   └── types
│       └── __init__.py      # Custom types and interfaces for the project
├── requirements.txt         # Lists dependencies for the project
└── README.md                # Documentation for the project
```

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd llmfarminf-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   Ensure you have an OpenAI API key and set it as an environment variable:
   ```
   export OPENAI_API_KEY='your-api-key'
   ```

## Usage
To generate new call transcripts and create a synthetic dataset, run the main application:
```
python src/main.py
```

## Dataset Generation
The application will utilize the `Llmfarminf` class to interact with OpenAI's API and generate call transcripts. The process is designed to create a dataset of approximately 50,000 entries, ensuring diversity and variability in the generated data.

## Contributing
Contributions to enhance the functionality or improve the dataset generation process are welcome. Please submit a pull request or open an issue for discussion.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.