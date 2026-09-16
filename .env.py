import os
def create_env_file():
    # Define your keys here
    # Remember to replace 'hf_your_actual_token_here' with the token you just generated!
    env_content = (
        "DEEPSEEK_API_KEY=your_deepseek_api_key_here\n"
        "GOOGLE_API_KEY=your_google_api_key_here\n"
        "HUGGINGFACEHUB_API_TOKEN=your_huggingface_hub_token_here\n"
    )
    
    # Write to the .env file in the current directory
    try:
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ .env file updated successfully!")
    except Exception as e:
        print(f"❌ Error updating file: {e}")

if __name__ == "__main__":
    create_env_file()