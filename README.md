
Project developed on Mac M2 Pro with an Anaconda envrionment and has not been tested on windows or other operating systems. Small adjustments to versions may be required. No GPU was used and should not be necessary.

Pytorch should be downloaded from:

https://pytorch.org/ - with the correct version for your operating system.

Setting Up the Environment:

To set up the environment and install the required packages for this project, follow these steps:

Clone the repository:

  git clone https://github.com/your-username/your-repository.git

Navigate to the project directory:

 cd to-repository

Create a new virtual environment:

 python -m venv myenv
Replace myenv with the desired name for your virtual environment.

* this may also be created using Anaconda - as with the original project.

Activate the virtual environment:

For Unix/Linux:
 source myenv/bin/activate

For Windows:
 myenv\Scripts\activate



Install the required packages:
 pip install -r requirements.txt
This command will install all the packages listed in the requirements.txt file.

That's it! You now have the environment set up and the necessary dependencies installed to run the project.


 ____________________________________

To generate data:

cd to /Data_generation and run main_gen.py

 To train and see val and test loss for a model:
 
 Replace the path to the data with actual paths and run **main.py** while in the root directory.

 

 
