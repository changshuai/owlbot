import os

def file_agent():
    """Simple file management agent"""
    context = []
    
    # Available capabilities
    capabilities = {
        'read_file': {
            'function': lambda path: open(path).read(),
            'description': 'Read file content'
        },
        'write_file': {
            'function': lambda path, content: open(path, 'w').write(content),
            'description': 'Write to file (overwrites)'
        },
        'list_dir': {
            'function': lambda path: os.listdir(path),
            'description': 'List directory contents'
        },
        'search_files': {
            'function': lambda query: [f for f in os.listdir('.') if query in f],
            'description': 'Search files by name'
        }
    }
    
    print("File Agent ready. Available capabilities:")
    for name, cap in capabilities.items():
        print(f"- {name}: {cap['description']}")
    
    # Simple agent loop
    while True:
        task = input("\nWhat should I do? (or 'exit' to quit): ")
        if task.lower() == 'exit':
            break
            
        # Here the model would normally decide which capability to use
        # For demo we'll directly map some commands
        if 'read' in task.lower() and 'file' in task.lower():
            path = input("File path: ")
            print("\nFile content:")
            print(capabilities['read_file']['function'](path))
        elif 'list' in task.lower() and 'files' in task.lower():
            path = input("Directory path (or . for current): ") or '.'
            print("\nDirectory contents:")
            print(capabilities['list_dir']['function'](path))
        else:
            print("I can help with file operations. Try asking to:")
            print("- Read a file\n- List files\n- Write to a file\n- Search files")

if __name__ == "__main__":
    file_agent()