## prerequisite: security boot has to be disabled  
sudo apt --purge autoremove nvidia*
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
apt-cache search nvidia | grep nvidia-driver-440
sudo apt-get install nvidia-driver-418
# sudo reboot
