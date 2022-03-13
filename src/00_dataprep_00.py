import tarfile
  
# open file
file = tarfile.open('../graphfile.tar')
  
# extracting file
file.extractall('../datasets')
  
file.close()
