function mount () {
    sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb
    sudo mkdir -p /mnt/disks/data
    sudo mount -o discard,defaults /dev/sdb /mnt/disks/data
    sudo chmod a+w /mnt/disks/data
    sudo cp /etc/fstab /etc/fstab.backup
    # export UUID=$(sudo blkid /dev/sdb | cut -f2 -d\    | cut -f2 -d\")
    # echo UUID=${UUID} /mnt/disks/data ext4 discard,defaults,NOFAIL_OPTION 0 2
    echo UUID=`sudo blkid -s UUID -o value /dev/sdb` /mnt/disks/data ext4 discard,defaults,NOFAIL_OPTION 0 2 | sudo tee -a /etc/fstab
    cat /etc/fstab
}

sudo apt install wget git screen neovim build-essential htop parallel sox libsox-fmt-mp3
wget https://gist.githubusercontent.com/ChrisWills/1337178/raw/8275b66c3ea86a562cdaa16f1cc6d9931d521e1b/.screenrc-main-example > ~/.screenrc
echo set -o vi >> ~/.bashrc

# git config --global user.name "YOUR USERNAME"
# git config --global user.email "YOUR EMAIL"
ssh-keygen -t rsa
cat ~/.ssh/id_rsa.pub
echo "add to https://github.com/settings/keys" before proceeding
read -n 1
git clone --recursive git@github.com:galv/lingvo-copy.git
git clone --recursive git@github.com:greg1232/mlcommons-speech.git
mount
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
conda env create -n lingvo -f lingvo-copy/environment.yml
echo "conda activate lingvo" >> ~/.bashrc