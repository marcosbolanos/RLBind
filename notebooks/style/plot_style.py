import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Font configuration - update to match transcriptomics exactly
font_path = './style/Nunito.ttf' 

# Add the font to the font manager
fm.fontManager.addfont(font_path)

# Configure matplotlib with proper font weights
plt.rcParams['font.family'] = 'Nunito'
plt.rcParams['font.weight'] = 'light'
plt.rcParams['figure.titleweight'] = 'medium'
plt.rcParams['axes.titleweight'] = 'semibold' 
plt.rcParams['axes.labelweight'] = 'light'