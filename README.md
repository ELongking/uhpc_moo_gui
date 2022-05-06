# Introduction
This simple program is implemented by our work, which is used to get high-accuracy ultra-high-performance concrete (or other concrete) mixtures-properties prediction model and back to do multi-objective optimization work at the same time. 

<details open>
<summary>Some features</summary>
  
- **Custom Settings**
  
  User can set their own restrictions and constraints casually, etc.
 
- **Universality**
  
  Different types of concrete data work well if meets the format requirements
  
- **Modifiability**
  
  If you know python, you can change the source code by yourself

This project is released under the [Apache 2.0 license](LICENSE).
</details>

# How to use
## Step 1
**First, prepare your data file and conduct the settings**

The first row is used to import your concrete data. Make sure that one process (dataflow) only support for one prediction pair. For example, A.csv contains mixtures     and compressive strength, B.csv contains mixtures and slump, etc. When you need more than two properties to get prediction model, please import and run the data separately. **Please see the specific format in document A**

The second row is used to choose to select the export location (folder) for the relevant results files. The files include the final prediction model file (.pkl), the log files during the process, and a file named bound.xlsx which will be used in step 2

The third row is optional. Like first row described, when you need to use multiple models generated in step 1 in advance in step 2, and the data used in the models generated in step 1 have different distributions, be sure to set this row. You need to concentate these whole data into a single .xlsx file. **Please see the specific format in document A**

**Second, run this program**

Be careful, the start button will only become clickable once you have set up the first two rows.

And wait the progress, you can check log for progress details. The program will alert you when all process have finished

## Step 2
**First, prepare your various files and conduct the settings**

The first row is used to import all the mixtures-properties prediction model files that you want to optimize (.pkl). Usually these model files are derived from the step 1. If you make the wrong adds, you can delete it by clicking the row and pressing the "remove" button on the right.

The second row is used to import user-defined file (.txt) containing various restrictions and constraints. This file include four parts
