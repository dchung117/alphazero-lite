# Download anaconda installer
wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh

# Install anaconda
shasum -a 256 Anaconda3*.sh
bash Anaconda3*.sh

echo "Installed Anaconda; run 'source ~/.bashrc; conda env create -f environment.yml' to complete setup."

# Delete install scripts
rm Anaconda3*.sh
