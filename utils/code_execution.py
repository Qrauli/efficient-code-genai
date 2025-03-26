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

def profile_with_scalene(code, config, test_input=None):
    """
    Profile the given code with Scalene for line-by-line performance metrics
    
    Parameters:
    code (str): Python code to execute
    config (Config): Configuration object
    test_input (str, optional): Input data for testing
    
    Returns:
    dict: Profiling results including line-by-line CPU time, memory usage, and allocation
    """
    # Create a temporary file for the code
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(code.encode())
    
    # Create a temporary file for output
    output_json_path = temp_file_path + '.json'
    
    try:
        # Set a shorter profiling interval for faster results with large datasets
        profile_interval = config.PROFILE_INTERVAL if hasattr(config, 'PROFILE_INTERVAL') else 4
        
        # Command to execute Scalene with JSON output and optimized settings
        cmd = [
            'python3', '-m', 'scalene',
            '--json', '--outfile', output_json_path,
            '--cpu', '--memory',
            '--profile-interval', str(profile_interval),  # Faster sampling
            '--reduced-profile',  # Lighter profiling for faster execution
            temp_file_path
        ]
        
        # Prepare input redirection if needed
        stdin = None
        if test_input:
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as input_file:
                input_file_path = input_file.name
                input_file.write(test_input.encode())
                stdin = open(input_file_path, 'r')
        
        # Use a shorter timeout for profiling than for regular execution
        profiling_timeout = min(5.0, config.TIMEOUT_SECONDS) if hasattr(config, 'TIMEOUT_SECONDS') else 5.0
        
        # Execute Scalene profiler
        process = subprocess.Popen(
            cmd,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        # Wait for the process to complete with timeout
        try:
            stdout, stderr = process.communicate(timeout=profiling_timeout)
            output = stdout.decode()
            error = stderr.decode()
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            output = stdout.decode()
            error = stderr.decode() + f"\nProfiling timed out after {profiling_timeout} seconds"
        
        # Load the JSON profile if it exists
        profile_data = {}
        if os.path.exists(output_json_path):
            with open(output_json_path, 'r') as f:
                import json
                try:
                    profile_data = json.load(f)
                except json.JSONDecodeError:
                    profile_data = {"error": "Failed to parse Scalene output JSON"}
        
        return {
            "profile_data": profile_data,
            "output": output,
            "error": error,
            "success": process.returncode == 0,
            "timed_out": "timed out" in error.lower()
        }
    
    except Exception as e:
        return {
            "profile_data": {},
            "output": "",
            "error": str(e) + "\n" + traceback.format_exc(),
            "success": False,
            "timed_out": False
        }
    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(output_json_path):
            os.unlink(output_json_path)
        if test_input and 'input_file_path' in locals() and os.path.exists(input_file_path):
            os.unlink(input_file_path)
            if stdin:
                stdin.close()