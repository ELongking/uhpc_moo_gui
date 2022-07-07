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

This project is released under the [Apache 2.0 license](./LICENSE).
</details>

# How to use
![image](https://github.com/ELongking/uhpc_moo_gui/blob/main/FC.png)
## Step 1
***First, prepare your data file and conduct the settings***

The first row is used to import your concrete data. Make sure that one process (dataflow) only support for one prediction pair. For example, A.csv contains mixtures     and compressive strength, B.csv contains mixtures and slump, etc. When you need more than two properties to get prediction model, please import and run the data separately. **Please see the specific format in [file](sample/tabular.xlsx)**

The second row is used to choose to select the export location (folder) for the relevant results files. The files include the final prediction model file (.pkl), the log files during the process, and a file named bound.xlsx which will be used in step 2

The third row is optional. Like first row described, when you need to use multiple models generated in step 1 in advance in step 2, and the data used in the models generated in step 1 have different distributions, be sure to set this row. You need to concentate these all data into a single .xlsx file by column (without properties in tabular).

***Second, run this program***

Be careful, the start button will only become clickable once you have set up the first two rows.

And wait the progress, you can check log for progress details. The program will alert you when all process have finished

## Step 2
***First, prepare your various files and conduct the settings (This part is very significant, which will directly affect your final optimization results)***

The first row is used to import all the mixtures-properties prediction model files that you want to optimize (.pkl). Usually these model files are derived from the step 1. If you make the wrong adds, you can delete it by clicking the row and pressing the "remove" button on the right.

The second row is used to import user-defined file (.txt) containing various restrictions and constraints. This file include four parts:

<div align="center">
  <b>Example tabular</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Cement</b>
      </td>
      <td>
        <b>Fly ash</b>
      </td>
      <td>
        <b>Silica fume</b>
      </td>
      <td>
        <b>Slag</b>
      </td>
      <td>
        <b>Limestone powder</b>
      </td>
      <td>
        <b>Water</b>
      </td>
      <td>
        <b>...</b>
      </td>
      <td>
        <b>Properties(for example strength)</b>
      </td>
    </tr>
  </tbody>
</table>

**First of all, be careful that if you want to represent a component column, for example like above, cement is located in the first column of excel tabular, you should use inverse(0) to represent it (start counting from 0)**

**For all sample please see [folder](sample)**

- Inequality: 

  The line used to express inequality. For example if you want to indicate that the water to cement ratio is less than 0.25 and more than 0.12, express equation should be *0.12<=inverse(5)/inverse(0)<=0.25*. Both ends of the inequality need to be a number.
  
  Be careful, we STRONGLY recommend that this section be as detailed as possible, because the algorithm itself does not have concept of objective scientific restrictions
  
- Bound:

  This line express the minimum and maximum values of the individual components. The minimum maximum value of each component is separated by ',' and the different components are separated by ';', the order should be consistent with the original data. For example: cement min,cement max;Fly ash min,Fly ash max...
  
- Prediction:
  
  This line express the minimum you want to get in prediction function. For example in practice, we usually want the final concrete strength is higher than a certain value. If you have N prediction model, you should use ',' to separate them, and if you dont need minimum bound, you should enter 'None' here.
  
- Customize:

  This line express additional user-defined optimization function (for example cost function). If the unit prices are respectively a,b,c,d,e, the equation should be *a\*inverse(0)+b\*inverse(1)+c\*inverse(2)+d\*inverse(3)+e\*inverse(4)*
  
- Standard:

  This line express the maximum and minimum values for the entire data distribution (exported from step 1 in bound.xlsx). Copy content from xlsx file directly but make sure that minimum values is in front.
  
The third row is used to choose to select the export location (folder) for the relevant results files. The files include the final mixtures optimization results, final corresponding properties and customized function results and log files during the process.

The final check box means, sometimes your constraints are too complex or contrary to mathematical foundation, and you cant aware of it. It may cause algorithm cant find the optimal solution set. If you still want to get a result under the previous constraint condition, you should turn the box to TRUE.

***Second, run this program***

Be careful, the start button will only become clickable once you have set up all the rows.

## Note
- Customized function now only support for solving the minimum value

- Only support for regression problem (the classification problem may lead to poor results)

- Everytime one process completed, you need re-open the whole program

## Future
There may be plans for continuous process improvement.

## Program packaging

Make sure you have installed pyinstaller if you have related python env.

And you need these following packages:

- pymoo
- scikit-learn
- numpy 
- pandas
- hyperopt

```Bash
pyinstaller -D -w --clean surface.py
```

And check dist folder. There we have packed the program, check the [link](./program_link)
