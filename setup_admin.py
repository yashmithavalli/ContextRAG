import os
import bcrypt
import yaml
import faiss

def main():
    print("AntiGravity Admin Setup")
    print("-" * 25)
    username = input("Enter admin username: ").strip()
    if not username:
        print("Username cannot be empty.")
        return
        
    password = input("Enter admin password: ").strip()
    if not password:
        print("Password cannot be empty.")
        return
        
    # Hash password with bcrypt
    salt = bcrypt.gensalt()
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    # Write entry to data/users.yaml
    os.makedirs("data", exist_ok=True)
    users_file = "data/users.yaml"
    users_data = {}
    
    if os.path.exists(users_file):
        with open(users_file, "r") as f:
            users_data = yaml.safe_load(f) or {}
            
    if "users" not in users_data:
        users_data["users"] = {}
        
    users_data["users"][username] = {
        "password_hash": hashed_pw,
        "role": "admin"
    }
    
    with open(users_file, "w") as f:
        yaml.dump(users_data, f)
        
    # Create data/index_{username}.faiss placeholder
    placeholder_path = f"data/index_{username}.faiss"
    if not os.path.exists(placeholder_path):
        index = faiss.IndexFlatL2(384)
        faiss.write_index(index, placeholder_path)
        
    print("✅ Admin user created. Run docker-compose up to start AntiGravity.")

if __name__ == "__main__":
    main()
