#!/usr/bin/env python3
"""
Hugging Face Login Helper
Provides multiple authentication methods for programmatic access
"""

import os
import getpass
from huggingface_hub import login, whoami


def login_with_token():
    """Login using a provided token"""
    print("=== Hugging Face Token Login ===")
    print("Get your token from: https://huggingface.co/settings/tokens")
    print("Make sure you have access to FLUX.1-schnell model")
    
    token = getpass.getpass("Enter your HF token: ")
    
    try:
        login(token=token)
        user_info = whoami()
        print(f"✅ Successfully logged in as: {user_info['name']}")
        return True
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return False


def login_with_env():
    """Set up environment variable login"""
    print("=== Environment Variable Setup ===")
    token = getpass.getpass("Enter your HF token (will be set as HF_TOKEN): ")
    
    os.environ["HF_TOKEN"] = token
    
    try:
        user_info = whoami()
        print(f"✅ Environment variable set. Logged in as: {user_info['name']}")
        print("Note: This token is only active for this session")
        return True
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False


def check_auth_status():
    """Check current authentication status"""
    try:
        user_info = whoami()
        print(f"✅ Currently logged in as: {user_info['name']}")
        return True
    except Exception:
        print("❌ Not currently authenticated")
        return False


def main():
    print("=== Hugging Face Authentication Helper ===")
    
    # Check if already authenticated
    if check_auth_status():
        return True
    
    print("\nChoose authentication method:")
    print("1. Interactive login (saves token permanently)")
    print("2. Token input (saves token permanently)")
    print("3. Environment variable (session only)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        try:
            login()
            return check_auth_status()
        except Exception as e:
            print(f"❌ Interactive login failed: {e}")
            return False
    elif choice == "2":
        return login_with_token()
    elif choice == "3":
        return login_with_env()
    else:
        print("❌ Invalid choice")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Ready to run FLUX tests!")
    else:
        print("\n❌ Authentication required before running FLUX tests")
