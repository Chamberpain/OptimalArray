from OptimalArray.Data.__init__ import ROOT_DIR
from OptimalArray.Utilities.Plot.__init__ import ROOT_DIR as PLOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from GeneralUtilities.Data.Filepath.search import find_files
from GeneralUtilities.Data.pickle_utilities import load

data_file_handler = FilePathHandler(ROOT_DIR,'OptimalArray')
file_handler = FilePathHandler(PLOT_DIR,'final_figures')


def make_filename(label,depth_idx,kk):
	return data_file_handler.tmp_file(label+'_'+str(depth_idx)+'_'+str(kk))

