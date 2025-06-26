import os
from openai import AzureOpenAI

class llmfarminf():
    def __init__(self):
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY", "ADD_YOUR_API_KEY_HERE")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://mha2c-mabd2a4o-eastus2.cognitiveservices.azure.com/")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
        )

    def _gen_message(self, sysprompt, userprompt):
        return [
            {"role": "system", "content": sysprompt},
            {"role": "user", "content": userprompt}
        ]

    def _completion(self, usertext, sysprompt):
        messages = self._gen_message(sysprompt, usertext)
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    obj = llmfarminf()
    prompt = "This is the DeepLearning for Final project. Please generate a response based on the system prompt."
    print(obj._completion(prompt, "You are a helpful assistant"))