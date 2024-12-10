Set-up Instructions:

1. Clone repository to your local machine:
mkdir ~/code/soulseekqt && cd "$_"
git clone git@github.com:soulseekqt/risky_predictive_policing.git
cd risky_predictive_policing

2. Then add raw_data folder:
mkdir ~/code/soulseekqt && cd "$_"


3. Install virtual environment on local machine:

pyenv virtualenv 3.10.6 risky_predictive_policing
pyenv local risky_predictive_policing

4. Include the Chicago Crimes dataset into raw_data' folder that was just created:
'chicago.csv'

Run install requirements:
pip install -r requirements.txt
