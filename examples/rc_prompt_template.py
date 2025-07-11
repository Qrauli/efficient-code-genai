
import json
import pandas as pd
import numpy as np

def _json_serial_helper(obj):
    """
    A helper function to make complex Python objects JSON serializable.
    This handles common types found in pandas DataFrames.
    """
    if isinstance(obj, (pd.Timestamp, pd.Period, pd.Timedelta)):
        return str(obj)
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        # Handle NaN specifically, as it's not valid JSON
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # This will catch datetime.date, datetime.datetime from the traceback
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    # For any other unhandled types, convert to string as a fallback
    # Let the default encoder raise the TypeError if it's something truly unhandled
    try:
        return str(obj)
    except Exception:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def summarize_dataset(dataset, n_samples):
    """Generate a summary of the dataset including column names, types, and a sample of data."""
    n = min(len(dataset), n_samples)
    sample = dataset.sample(n=n, random_state=42) if n > 0 else dataset.head(0)
    summary = {
        "columns": list(sample.columns),
        "dtypes": {col: str(dtype) for col, dtype in dict(sample.dtypes).items()},
        "sample": sample.head().to_dict(orient="records"),
    }

    # Convert the summary dictionary to a JSON-formatted string using the helper
    return json.dumps(summary, indent=4, default=_json_serial_helper)

def generate_code_prompt():
    """
    Generates a prompt for the language model to create a Python function.
        
    Returns:
    -------
    str
        The generated prompt.
    """

    basic_instruction = '''
    You are tasked with generating a Python program for data quality assurance. Your goal is to create a function that
    applies a specified rule to a dataset represented as a pandas DataFrame.

     ### Details:
    - The rule is "{rule.rule}"
    - The explanation of the logic and rationale behind the rule is "{rule.explanation}"
    - The dataset schema and sample data are provided to help you understand the structure and typical values.

    ### Sample Data:
    {sample}

    ### Task:
    - Write a function named `{fun_name}` that processes the DataFrame according to the given rule. 
    
    '''

    further_instruction = '''
    ### Task Details:
    
    Before performing further actions, first determine whether the given rule is a **single-row rule** or a **multi-row rule**. 
    Use the definitions below:

    1. **Single-Row Rule**:  
       A rule that evaluates each row independently, ensuring that the values within the same row satisfy certain conditions or constraints. 
       These rules do not depend on relationships with other rows.  

    2. **Multi-Row Rule**:  
       A rule that evaluates relationships, dependencies, or patterns across multiple rows in the dataset. 
       These rules involve comparisons, groupings, or validations involving more than one row.
    
    
    The generated function should:
    
    I. compute Confidence and Support of each rule.
    II. generate `satisfactions`, `violations` : presentations of rows or row groups that satisfy or violate the rule. This presentation should utilize the index of the dataframe to represent rows. 
       
    - The function should return a dictionary with keys:
        - `support`: the support value   
        - `confidence`: the confidence value.
        - `satisfactions`, `violations` 
        
   
    For single-row rules, the `satisfactions` and `violations` should be represented as two seperated sets:
    <row_index_set>, # Each set row_index_set contains the identifiers (i.e., dataframe index of rows) for the rows that either satisfy or violate the rule.
    
    For multi-row rules, first check if the rule is a **group-validation rule** defined as:
    
    Group-Based Validation Rules involve grouping rows based on specific column values and evaluating whether certain criteria are satisfied within each group. 
    Examples of these rules include functional dependencies (FDs), unique key constraints, outlier detection, aggregation constraints, and more. 
    
    If the rule is not a group-validation rule, return `None`, as it is currently beyond the system's capabilities. 
    If the rule is a group-validation rule, proceed by following the instructions below.
    
    '''

    uniform_representation = """
    
    ### How to represent `satisfactions` and `violations` for multi-row rules:
    
    A data quality checking rule that applies to multiple rows in each check should have its **violations** and **satisfactions** represented as a dictionary:
    
    {
        group_key: 
            a list of dictionaries  
            or  
            a single dictionary
    }
    
    Each group_key represents a group of rows, i.e., set of rows grouped by one or more columns. Each group_key can be structured as:
    - A single tuple if the group key is defined on a single column, for example: (employee_id, "tech112212").
    - A tuple of key-value pairs if the group key is defined on multiple columns: ((team_id, "sales"), ("employee_name", "Jack")).
    
    Each **group_key’s corresponding value** represents either **violations** or **satisfactions** of the rule in that group. 
    This value:
    - Can be a **list of dictionaries** (when multiple violation representations or satisfaction representations exist in the group).  
    - Or a single dictionary (when a single violation representation or a single satisfaction representation is required.)
    
    Each dictionary inside the corresponding value of a group_key is called a **violation representation** or a **satisfaction representation**.  
    
    Each representation dictionary consists of keys that describe the **roles played by different rows (or by different row groups)** in the violation or satisfaction.  
    The **values** corresponding to these keys store the **row indexes** involved in the representation.
    
    In each representation dictionary, a key’s corresponding value can be:
      - A **single row index** (if only one row plays that role).
      - A **list of row indexes** (if multiple rows play that role).
    
    By default, each distinct role in a representation forms a **"participant"**—a set of rows or a single row that contribute to the rule violation or satisfaction in a specific way.
    
    - If there are multiple roles, then the **participants of different roles together explain** why the rule is **violated or satisfied**.
    - If there is only one role, then the rows assigned to that role collectively form a **participant** that explains how the rule is **violated or satisfied**.
    
    Within each **group**, all participants together explain **why the rule is violated or satisfied** in that group.
    
    ---
    
    ## **Examples of Rule Violation and Satisfaction Representations**  
    
    ### **Example 1: Sum-to-Total Comparison Rule**
    **Rule Definition:**  
    After grouping rows by columns **project_group** and **project_name**, the sum of the **"amount"** of rows where **"amount type"** is `"sub-part"`  
    must be **equal to** the **"amount"** of the row where **"amount type"** is `"total"`.  
    
    #### **Violation Representation:**
    violations = {
        (("project_group", "technique"), ("project_name", "DeepClean")): {
            "sum": [1, 6, 9],  # Violation Participant 1 in this group
            "compare": 11      # Violation Participant 2 in this group
        },  
        (("project_group", "engineering"), ("project_name", "FastTrac")): {
            "sum": [11, 16, 29],  
            "compare": 33  
        }  
    }
    
    #### **Satisfaction Representation:**
    satisfactions = {
        (("project_group", "marketing"), ("project_name", "AdBoost")): {
            "sum": [7, 14, 18],  # Satisfaction Participant 1
            "compare": 20        # Satisfaction Participant 2
        }
    }
    
    #### **Explanation:**
    Each **group key’s value is a single dictionary**, indicating that each group contains only **one representation** (violation or satisfaction).  
    Within each representation:
    - `"sum"` → Rows whose **amounts are summed** (**Participant 1**).
    - `"compare"` → Row containing the **total amount** (**Participant 2**).  
    
    A **violation** occurs when the sum does **not** match the total amount, while a **satisfaction** occurs when the sum **correctly** matches the total.
    
    ---
    
    ### **Example 2: Sum-to-Threshold Comparison Rule**
    **Rule Definition:**  
    After grouping rows by columns **team_id** and **employee_name**, the sum of the **"salary"** of three months in a quarter  
    must **not exceed** 100,000.  
    
    #### **Violation Representation:**
    violations = {
        (("team_id", "sales"), ("employee_name", "Jack")): [
            {"sum": [22, 23, 24]},  # Violation Representation 1
            {"sum": [25, 26, 27]}   # Violation Representation 2
        ],  
        (("team_id", "HR"), ("employee_name", "Jane")): {
            "sum": [100, 101, 102]  # Single violation representation
        }  
    }
    
    #### **Satisfaction Representation:**
    satisfactions = {
        (("team_id", "engineering"), ("employee_name", "Alice")): {
            "sum": [30, 31, 32]  # Single satisfaction representation
        }
    }
    
    #### **Explanation:**
    - The first **group key** has **multiple violation representations** (a list of dictionaries), meaning multiple violations exist for `"Jack"` in the `"sales"` team.  
    - The second **group key** has **a single satisfaction representation**, meaning `"Alice"` in `"engineering"` **satisfies the rule**.
    
    Each representation consists of a single role:
    - `"sum"` → Rows whose **salary is summed**.
    
    Each **violation participant** exceeds the **100,000 threshold**, while each **satisfaction participant** stays **within the limit**.
    
    ---
    
    ### **Example 3: Unique Key Constraint Rule**
    **Rule Definition:**  
    No two rows should have the **same values** in columns **A, B, and C** (i.e., they must form a unique key).  
    
    #### **Violation Representation:**
    violations = {
        (("A", "100"), ("B", "200"), ("C", "300")): {
            "duplicates": [1, 2, 3]  
        },  
        (("A", "400"), ("B", "500"), ("C", "600")): {
            "duplicates": [4, 5, 6]  
        }  
    }
    
    #### **Satisfaction Representation:**
    satisfactions = {
        (("A", "700"), ("B", "800"), ("C", "900")): {
            "unique_record": 7
        } , 
       (("A", "1000"), ("B", "800"), ("C", "2900")): {
            "unique_record": 8
        } ,
     (("A", "1000"), ("B", "800"), ("C", "2900")): {
            "unique_record": 9
        } ,
        ... ...
         
    }
    
    #### **Explanation:**
    Each **group key’s value is a single dictionary**, meaning there is only **one representation per group**.  
    Each representation consists of a single role:
    - `"duplicates"` → Rows that **violate** uniqueness.
    - `"unique_record"` → A row that **satisfy** uniqueness.
    
    Each **violation set** contains duplicates, while each **satisfaction set** confirms unique records.
    
    
    ### ** Example 4: Monotonic Amount Rule **
    ** Rule Definition:  **
    After grouping rows by **department_id**, if an **employee A** has a **higher position** than **employee B**,  
    then **A's salary should be higher than B’s**.  
    
    If any row representing an employee with a higher position has a **salary that is not strictly greater** than the salary of another row representing an employee with a lower position, then the rule is **violated**.
    
    #### Violation Representation:
    violations = {
        ("department_id", "business"): [
            {
               "row with higher position": 3, 
               "row with lower position": 5
           },  # Violation Representation 1 in this group
           
           {
               "row with higher position": 6, 
               "row with lower position": 10
           },  # Violation Representation 2 in this group
        ],  
        ("department_id", "tech"): [
          {
               "row with higher position": 20, 
               "row with lower position": 25
           },        
           {
               "row with higher position": 26, 
               "row with lower position": 27
           },      
         ]  
    }
    
    #### Satisfaction Representation:
    satisfactions = {
        ("department_id", "finance"): [
            {
               "row with higher position": 12, 
               "row with lower position": 18
            },  # Satisfaction Representation 1 in this group
            
            {
               "row with higher position": 14, 
               "row with lower position": 19
            }   # Satisfaction Representation 2 in this group
        ]
    }
    
    
    #### Explanation:
    Each **group key** represents a **department** where the rule is checked.  
    - A **group key’s value** is a **list of dictionaries**, meaning multiple comparisons exist.  
    - Each representation consists of:
      - `"row with higher position"` → The employee who should have a **higher salary**.
      - `"row with lower position"` → The employee who should have a **lower salary**.
    
    #### Rule Outcome:
    - **Violations occur** when an employee with a higher position **does not** have a strictly higher salary.
    - **Satisfactions occur** when the expected salary hierarchy is **correctly maintained**.
    
    
    
    ### **Example 5: Functional Dependency (FD) Rule**
    **Rule Definition:**  
    If two rows have the **same values** for key columns (e.g., "department", "employee_id"),  
    then they **must have the same value** for a dependent column (e.g., "empleyee_name").  
    
    #### **Violation Representation:**
    ```python
    violations = {
        (("department", "tech"), ("employee_id", "e12121")): {
            ("employee_name", "Jacky Ma"): [12, 15, 19], 
            ("employee_name", "Leon Ma"):  [20, 22, 28],
            ("employee_name", "Leon Maha"):  [39, 42, 50]
         },
        
         (("department", "marketing"), ("employee_id", "m121378")): {
            ("employee_name", "Luna Wilson"): [101, 102, 103], 
            ("employee_name", "Luna William"):  99,
            ("employee_name", "Luna Ma"):  [91, 92]
         }
    }
    ```
    
    #### **Satisfaction Representation:**
    ```python
    satisfactions= {
        (("department", "tech"), ("employee_id", "e292122")): {
            ("employee_name", "Yiyang Qianxi"): 122,      
        },
        
         (("department", "marketing"), ("employee_id", "m99921")): {
            ("employee_name", "Yuqi Song"): [901, 902, 903, 904, 905]     
         }
    }
    ```
    
    #### **Explanation:**
    Each **group key** represents a unique combination of key columns (`department`, `employee_id`), where the rule ensures that the dependent column (`employee_name`) must have consistent values.
    
    - **Violation representation**: 
      - If multiple **different** values appear in the `employee_name` field for the **same** (`department`, `employee_id`) group, the functional dependency is violated.
      - Each **violation participant** represents a **distinct employee_name** that appears in conflict within the same group.
      - Example:  
        - Employee ID `"e12121"` in the `"tech"` department appears with three different names: `"Jacky Ma"`, `"Leon Ma"`, and `"Leon Maha"`, violating the functional dependency.
        - Employee ID `"m121378"` in the `"marketing"` department appears with `"Luna Wilson"`, `"Luna William"`, and `"Luna Ma"`, indicating a similar violation.
    
    - **Satisfaction representation**:
      - If all rows within a (`department`, `employee_id`) group have **only one unique value** for `employee_name`, then the functional dependency is satisfied.
      - Each **satisfaction participant** represents a **single employee_name** that correctly follows the dependency.
      - Example:
        - Employee ID `"e292122"` in the `"tech"` department consistently appears as `"Yiyang Qianxi"`, ensuring the dependency is met.
        - Employee ID `"m99921"` in the `"marketing"` department is consistently recorded as `"Yuqi Song"`, satisfying the rule.
    
    Each **group key’s value** is a dictionary where:
    - The **keys** represent distinct values found in the `employee_name` column.
    - The **values** are row indexes indicating where each `employee_name` occurs.
    
    
    ---
    
    ## **Summary of Structure Consistency**
    1. **Each group key** represents a unique set of rows grouped by column values.
    2. **Each group key’s value** (representations) can be:
       - A **single dictionary** (one representation).
       - A **list of dictionaries** (multiple representations).
    3. **Each representation**:
       - Defines roles that rows play in the violation or satisfaction.
       - Stores row indexes corresponding to each role.
       - The combination of **participants** explains why the rule is **violated or satisfied**.

"""

    calculation = """
    
    ### How calculate `confidence` and `support`:
    
    #### for **Single-Row Rule**
    if the rule is a **Single-Row Rule**, then:
    
    - Support = (Number of rows where the rule is applicable, i.e., the number of rows the rule was checked on) / (Total number of rows)
    
    - Confidence = (Number of rows where the entire rule is satisfied) / (Number of rows where the rule is applicable, i.e., the number of rows the rule was checked on)

    #### for **Multi-Row Rule**
    if the rule is a **Multi-Row Rule**, then follow the following instructions:
    
    ### 1. Definition of Confidence
    **Confidence** measures how often the rule is satisfied when applied to the dataset. It is calculated as:
    
    Confidence = (Number of groups in satisfactions) / (Number of groups in violations + Number of groups in satisfactions)
    
    Note that number of groups means number of `group_key`s
    
    #### Steps to Compute Confidence:
    1. **Count the number of group keys in the `satisfactions` dictionary**.
    2. **Count the number of group keys in the `violations` dictionary**.
    3. **Apply the formula** using these counts.
    
    ---
    
    ### 2. Definition of Support
    **Support** measures how many rows in the dataset are involved in the rule, relative to the total number of rows in the dataset. It is calculated as:
    
    Support = (Number of unique rows involved in satisfactions and violations) / (Total number of rows in the dataset)
    
    #### Steps to Compute Support:
    1. **Extract all row indexes** that appear in **both `violations` and `satisfactions`**.
    2. **Count the number of unique row indexes**.
    3. **Divide this count by the total number of rows in the dataset**.
    
    ---
    
    ## 3. Example Calculation
    
    ### Example Data
    violations = {
        (("project_group", "technique"), ("project_name", "DeepClean")): {
            "sum": [1, 6, 9],
            "compare": 11
        },  
        (("project_group", "engineering"), ("project_name", "FastTrac")): {
            "sum": [11, 16, 29],
            "compare": 33
        }
    }
    
    satisfactions = {
        (("project_group", "marketing"), ("project_name", "AdBoost")): {
            "sum": [7, 14, 18],
            "compare": 20
        }
    }
    
    total_rows = 100  # Example: The dataset has 100 rows
    
    ### Step 1: Compute Confidence
    - Number of groups in `satisfactions` = **1** (`AdBoost`)
    - Number of groups in `violations` = **2** (`DeepClean`, `FastTrac`)
    
    Confidence = 1 / (1 + 2) = 1 / 3 = 0.33 (33%)
    
    ### Step 2: Compute Support
    - Unique row indexes from `violations`: {1, 6, 9, 11, 16, 29, 33} (7 unique rows)
    - Unique row indexes from `satisfactions`: {7, 14, 18, 20} (4 unique rows)
    - Total unique row indexes involved: {1, 6, 9, 11, 16, 29, 33, 7, 14, 18, 20} (11 unique rows)
    
    Support = 11 / 100 = 0.11 (11%)
    
    ---
    """

    notes = '''
    ### Important:
    - Ensure that in the documentation of the function, the original rule is specified. 
    - Ensure that all necessary libraries are imported at the beginning of the code. For example, `import pandas as pd`.
    - Do not include any example usage or extra code outside the function definition.
    - In the generated code, please delete unused intermediate variables to free memory before returning the results.
      Use `del` to delete variables and `gc.collect()` to free memory.
    - In the generated code, you should first limit the input DataFrame to the columns used,
      eliminating unused columns, and use this simplified DataFrame for the remaining operations.
      
    - [IMPORTANT EFFICIENCY CONSIDERATION] 
      Note that the code will be used for tables with multiple millions of rows. 
      Ensure that the code is efficient and uses as little memory as possible.
      Try to avoid copying and use in-place operations if possible.
      
    - [IMPORTANT] If the rule is not about checking None values, please exclude all rows where any of the relevant columns is None, using df.dropna(subset=relevant_columns), before analysis.
    
    - [IMPORTANT] Be careful when you write the code of generating the inner dictionaries in the violations, 
        do not replace the existing entries already contained in the inner dictionary. 
        For example, if the current violations is 
        violations = {{
                (("A", "100"), ("E", "200")): {{
                    (("B", "high"), ("C", "nice")): [1, 2, 3],
                }}
            }}
        
        and your code wants to insert (("B", "low"), ("C", "poor")): [4, 5], 
        the new violations should become:
        violations = {{
                (("A", "100"), ("E", "200")): {{
                    (("B", "high"), ("C", "nice")): [1, 2, 3],
                    (("B", "low"), ("C", "poor")): [4, 5]
                }}
            }}
        but not:
        violations = {{
                (("A", "100"), ("E", "200")): {{
                    (("B", "low"), ("C", "poor")): [4, 5]
                }}
            }}
    
    - [IMPORTANT] when iterating a dictionary such as grouped, such as calculating the total numbers of rows in all groups, avoid mistakes scuh as follows:
        
        # grouped is a dictionary with group_key and the corresponding group
        # below is an incorrect line
        fully_compliant_rows = sum(len(group) for group in grouped if len(group) == 1)
        
        The correct line should be:
        fully_compliant_rows = sum(len(group) for group_key, group in grouped.items() if len(group) == 1)
        
    - [IMPORTANT] When the rule is conditional and includes specific conditions, the calculations for satisfactions and violations should focus exclusively on the units that meet these conditions.
    - [IMPORTANT] Ensure that each key used in the dictionaries for satisfactions or violations of group validation rules, including keys in both outer and inner dictionaries, always includes the column name when a column value is part of the key. For example, valid keys can be ("A", 100) or (("A", 100), ("B", 200)), but not (100) or (100, 200).
        # Below is an incorrect key:
        violations_dict = dict()
        satisfactions_dict = dict()
        for group_key, group in grouped:
            violations_dict[group_key] = ....
            satisfactions_dict[group_key] = ..

        # The correct key should be:
        violations_dict = dict()
        satisfactions_dict = dict()
        for group_key, group in grouped:
            key = ((group_column[0], group_key[0]), (group_column[1], group_key[1]), ..., (group_column[n], group_key[n]))
            # Note: If the group has a single column, the key should be:
            # key = ((group_key, group_column[0]),)
            violations_dict[key] = .....
            satisfactions_dict[key] = .....

    

     - [IMPORTANT] For the inner key (<DETAIL_KEY_1>, <DETAIL_KEY_2>, ...), whenever possible, use the rule head column-value pairs tuple.

        Example 1: If we have a functional dependency X -> Y, the inner key will look like:
        ("Y", some_value)

        Example 2: If we have a rule Z -> （A， B）, the inner key will look like:
        (("A", 120), ("B", 110))
    

    - [IMPORTANT] The index of a row should refer to the DataFrame's actual index, not a column named "index" even if such a column exists.

    '''

    output_spec = '''
    ### Output Specification:
     Your response should ONLY contain the Python function wrapped in ```python and ``` markers.
    '''

    common_mistakes = """
    
    ### Note:
    
    Common Mistakes to Avoid When You Generate Code:
    
    1. Failing to Prepare Data for Regular Expressions: Not converting data to the correct format (e.g., strings) before applying regular expressions can lead to errors or unintended matches.
    2. Ignoring Missing Values: Failing to account for NaN values in the dataset can lead to unexpected results. Always handle missing values using fillna() or dropna() appropriately.
    3. Using Chained Indexing: Using chained indexing (e.g., df[df['column'] > 0]['column2'] = value) can lead to SettingWithCopy warnings and incorrect assignments. Use .loc[] instead.
    4. Iterating Inefficiently: Using .iterrows() or .apply() for row-wise operations instead of vectorized operations can significantly reduce performance.
    5. Allowing Data Type Mismatches: Not ensuring columns have consistent data types before operations can cause errors. Use df.astype() to cast types where needed.
    6. Creating Unnecessary Copies: Creating unnecessary copies of the dataframe wastes memory. Work on slices or views whenever possible.
    7. Mismanaging Index Operations: Resetting or setting indexes carelessly can disrupt the integrity of the dataframe. Always verify the dataframe after index operations.
    8. Assuming Column Order is Fixed: Relying on column order instead of column names can break the code if the order changes unexpectedly.
    9. Overwriting Critical Data: Overwriting important variables inadvertently can cause loss of critical data. Use meaningful variable names to track changes.
    10. Ignoring Memory Usage: Processing large datasets without monitoring memory usage can lead to crashes. Use df.info() and optimize operations for memory efficiency.
    11. Recalculating Intermediate Results: Recomputing the same results multiple times instead of storing them in temporary variables wastes resources.
    12. Hardcoding Values: Hardcoding column names, thresholds, or parameters reduces flexibility. Use variables or configuration files instead.
    13. Ignoring Performance Warnings: Overlooking warnings or errors in the console may result in performance or correctness issues.
    14. Skipping Code Documentation: Failing to add comments for non-trivial operations makes the code harder to understand and maintain.
    15. Neglecting Thorough Testing: Not testing the code with edge cases like empty dataframes, extreme values, or unexpected structures can result in undetected bugs.
    16. Overlooking Duplicate Entries: Ignoring duplicate rows or entries can affect data quality checks. Use df.duplicated() to identify and handle duplicates.
    17. Using Inconsistent Column Name Case: Referencing column names inconsistently with case sensitivity can cause KeyErrors in datasets with varying conventions.
    18. Failing to Validate Output: Not verifying that data quality rules are applied correctly can result in undetected errors. Check flagged rows and passing rows.
    19. Altering Indices Unintentionally: Changing indices during intermediate steps without tracking the original index can cause alignment issues.
    20. Misusing Inplace Operations: Using inplace=True carelessly can overwrite data unintentionally. Avoid inplace operations unless absolutely necessary.
    """

    special_note = """
    
    ### [!important] Special Note:

    #### Please strictly stick to the following coding convention relevant to tuples:

    When specifying a tuple in the code, please adhere to the following steps:

    1. **Define each element individually**:
    for example:
        e1 = "AAA"
        e2 = "BBB"

    2. **Construct the tuple explicitly**:
    for example:
        group_tuple = (e1, e2)

    **Important:** 
    - Always use the explicit tuple syntax `(e1, e2)`. 
    - DO NOT use the `tuple()` constructor, such as `tuple(e1, e2)` -- this is not allowed ! 
    
    """

    return basic_instruction + further_instruction + uniform_representation \
           + calculation + notes + output_spec + common_mistakes + special_note


# FIXME before reusing this function, it should be changed according to the propmpt the above function
def fix_bug_prompt():
    """
    Generates a prompt for the language model to fix a Python function with an error.

    Returns:
    -------
    str
        The generated prompt.
    """
    raise NotImplementedError(
        'this is an outdated function, it should be rewritten before used. '
    )
    prompt = """
    A Python program was generated to apply a data quality assurance rule.
    The goal was to create a function named `{fun_name}` that processes a pandas DataFrame according to a specified rule.
    The function should:
        - Compute Support: The proportion of rows where the body of the rule is satisfied. If the rule has no body, support is 1.
        - Compute Confidence: The proportion of rows where both the body and head of the rule are satisfied, out of the rows where the body is satisfied.
        - Return:
            - The support and confidence values.
            - The row indexes (from the original DataFrame) of either the violating rows or the satisfying rows, but only one of them, as they can be inferred from each other.
                          **The row indexes should be returned as a `set` object for performance reasons.**
            - A boolean indicator (`is_violations`):
                - `True` indicates that the returned indexes are of the violating rows.
                - `False` indicates that the returned indexes are of the satisfying rows.
            - The function should decide which indexes to return based on the confidence value; for example, if the confidence is high (e.g., 0.9 or above), returning `violating_rows_indexes` would be more efficient since it is typically much smaller than `satisfying_rows_indexes`.
    However, an error occurred during execution. Your task is to analyze the error and provide a corrected version of the program.
    Sample Data:
    {sample}
    Error Message:
    {error}
    Important:
        - Ensure all necessary libraries are imported at the beginning of the code, such as `import pandas as pd`.
        - Do not include any example usage or extra code outside the function definition.
        - Ensure that the code is efficient and uses as little memory as possible.
          Try to avoid copying and use in-place operations if possible.
        - In the generated code, please delete unused intermediate variables to free memory before returning the results.
          Use `del` to delete variables and `gc.collect()` to free memory.
        - In the generated code, you should first limit the input DataFrame to the columns used,
          eliminating unused columns, and use this simplified DataFrame for the remaining operations.

    Output Specification:
        - Return the results in a valid JSON format with the key "program" containing the function code.
        - The code should be a single-line string with escaped newlines and quotes.
        - Example of expected JSON format:
    {{
        "program": "import pandas as pd\\ndef {fun_name}(dataframe):\\n    # Implement the logic here\\n    ...\\n    return support, confidence, row_indexes, is_violations"
    }}
"""
    return prompt
