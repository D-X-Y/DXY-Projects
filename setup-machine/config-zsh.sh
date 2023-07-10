#!/bin/bash
echo 'current system : ' `uname`

chsh -s /bin/zsh
echo 'change the bash to zsh'
sh -c "$(curl -fsSL https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
echo 'install the zsh done'
echo '--------------------'

# Install coreutils to have some commands in the GNU version
brew install coreutils

if [[ `uname` == 'Darwin' ]]; then
  echo 'Mac osx copy zshrc.linux'
  cp zshrc.mac ~/.zshrc
else
  echo 'Linux copy zshrc.linux'
  cp zshrc.linux ~/.zshrc
fi



# Install fonts and colors

mkdir -p ~/.dir_colors
git clone https://github.com/seebi/dircolors-solarized.git
cp dircolors-solarized/dircolors.* ~/.dir_colors/
rm -rf dircolors-solarized

mkdir -p ~/.local
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git
cp -r zsh-syntax-highlighting ~/.local
rm -rf zsh-syntax-highlighting

if [[ `uname` == 'Darwin' ]]; then
  echo 'Install the Powerline fonts for Mac Osx'
  git clone https://github.com/powerline/fonts.git
  cd fonts
  ./install.sh
  cd ..
else
  echo 'Install the Powerline fonts for Linux (Skip)'
  #conda install powerline
fi
