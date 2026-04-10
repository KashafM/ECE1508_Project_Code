# need to create a ~/.synapseConfig file in users home directory with authorization token

import synapseclient
import shutil
import glob

syn = synapseclient.Synapse()
syn.login()
entity = syn.get("syn60086071", downloadLocation='./data')
print("Extracting Files...")
shutil.unpack_archive(glob.glob('./data/*.zip')[0], './dataset')
shutil.rmtree('./data')
print("Dataset downloaded!")