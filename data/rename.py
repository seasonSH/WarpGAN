import os


prefix = 'data/WebCaricature/OriginalImages'

for oldname in os.listdir(prefix):
    newname = oldname.replace(' ', '_')
    os.rename(os.path.join(prefix, oldname), os.path.join(prefix, newname))

    
