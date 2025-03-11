import time
import traceback
import subprocess
import tempfile
import os
import psutil  # You may need to install this: pip install psutil
import signal
import platform

def indent_code(code, indent='    '):
    """Properly indent code with the specified indent string"""
    # Split the code into lines
    lines = code.split('\n')
    # Add the indent to each non-empty line
    indented_lines = [indent + line if line.strip() else line for line in lines]
    # Join the lines back together
    return '\n'.join(indented_lines)

def execute_code(code, config, test_input=None):
    """
    Execute the given code in a safe environment and measure performance metrics
    
    Parameters:
    code (str): Python code to execute
    config (Config): Configuration object
    test_input (str, optional): Input data for testing
    
    Returns:
    dict: Execution results including time, memory usage, and output
    """
    # Create a temporary file for the code
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(code.encode())
    
    # Create another temporary file for test input if provided
    input_file_path = None
    if test_input:
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as input_file:
            input_file_path = input_file.name
            input_file.write(test_input.encode())
    
    try:
        # Command to execute
        cmd = ['python', temp_file_path]
        
        # Prepare input redirection if needed
        stdin = None
        if input_file_path:
            stdin = open(input_file_path, 'r')
        
        # Measure execution time
        start_time = time.time()
        
        # Execute the code as a subprocess
        process = subprocess.Popen(
            cmd,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        # Monitor memory usage during execution
        max_memory = 0
        p = psutil.Process(process.pid)
        process_terminated = False
        
        while process.poll() is None:
            try:
                # Get memory info in MB
                memory_info = p.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                max_memory = max(max_memory, memory_mb)
                
                # Check if memory limit exceeded
                if memory_mb > config.MAX_MEMORY_MB:
                    process.kill()
                    process_terminated = True
                    break
                
                # Check if timeout exceeded
                if time.time() - start_time > config.TIMEOUT_SECONDS:
                    process.kill()
                    process_terminated = True
                    break
                
                time.sleep(0.01)  # Small sleep to reduce CPU usage of monitoring
            except:
                # Process may have ended
                break
        
        # Get output and errors
        try:
            stdout, stderr = process.communicate(timeout=0.5)  # Short timeout for already terminated processes
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Decode output
        output = stdout.decode()
        error = stderr.decode()
        
        # Determine success
        success = process.returncode == 0 and not process_terminated
        
        if process_terminated:
            if memory_mb > config.MAX_MEMORY_MB:
                error = f"Memory limit exceeded ({memory_mb:.2f} MB used, limit: {config.MAX_MEMORY_MB} MB)"
            else:
                error = f"Execution timed out (limit: {config.TIMEOUT_SECONDS} seconds)"
        
        return {
            "execution_time": execution_time,
            "memory_usage": max_memory,  # Now we return actual memory usage in MB
            "output": output,
            "error": error,
            "success": success
        }
    
    except Exception as e:
        return {
            "execution_time": None,
            "memory_usage": None,
            "output": "",
            "error": str(e) + "\n" + traceback.format_exc(),
            "success": False
        }
    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if input_file_path and os.path.exists(input_file_path):
            os.unlink(input_file_path)
            if stdin:
                stdin.close()