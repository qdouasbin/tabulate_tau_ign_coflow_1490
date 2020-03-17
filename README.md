# How to use the scripts

> Always copy the original hdf5 file and work from the copy instead of the 
> original file (to avoid regenerating the table each time something goes wrong)


 1. Replace names datasets by the ones in the cantera xml
    - `changeTable2XMLspecies.py`
 2. Read table nodes and write dictionaries as binary files
    - `ReadTableNodes.py`
 3. Compute the ignition time for each of the nodes of the FPVA table
    - `tau_ignition_table_nodes.py`
 4. Compute/plot gradients
    - `plot_res_tau_ign_table.py`
 5. Add the derivative to the HDF5 table
    - `AddDerivative2Table.py`
 6. Transform hdf5 tabke to Tecplot with CharlesX tool and look at it


 > Careful, the FPV table (.h5) are too large to be contained in this repository.
