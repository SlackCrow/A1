#ifndef A1_HPP
#define A1_HPP

#include <vector>
#include <map>

template <typename T, typename Hash>
void mpi_shuffle(std::vector<T>& inout, Hash hash, MPI_Datatype Type, MPI_Comm Comm) 
{   
    std::map<int,std::vector<T>> tempMap;
    std::vector<T> tempSave;

    int nodeSize;
    MPI_Comm_size(MPI_COMM_WORLD, &nodeSize); // get the total # of ranks
    int currentRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &currentRank); // get the rank for the current PE

    // hash the vector of t given with the hash funciton given and store it in a map
    for(auto x : inout) // iterate through the input vector
    {
        int hashedValue = hash(x); // hash x using the hash function given
        int hashLoc = hashedValue % nodeSize; // modulo of the value we got to get the hash table loc which here dictates which rank it belongs to.
        if(hashLoc == currentRank) // if the modulo value == the current rank, save the value in the temp vector;
        {
            tempSave.emplace_back(x);
        }
        else // check if the modulo value is > than the rank of the last processor 
        {
            tempMap[hashLoc].emplace_back(x);
        }
    }

    // gather map to every PE
    for(int i = 0; i < nodeSize; ++i)  
    {
        int *toRecvSize; 
        if(currentRank == i)  // only initialize when current rank == i
            toRecvSize = new int[nodeSize];
        int localdata =  tempMap[i].size(); 
        MPI_Gather(&localdata, 1, MPI_INT, toRecvSize, 1, MPI_INT, i, Comm); // every PE sends the size of the data that is going to be sent to rank i.
        T *totalData;
        int *disps; 
        int total = 0; // the size of the data from every PE combined
        if(currentRank == i) // only initialize when current rank == i
        {
            disps = new int[nodeSize];
            for(int j = 0; j < nodeSize; ++j)
                total += toRecvSize[j]; // calculate the total size of the final array
            totalData = new T[total]; 
            for (int j = 0; j < nodeSize; ++j) 
                disps[j] = (j > 0) ? (disps[j-1] + toRecvSize[j-1]) : 0; // calculate displacement
        }
        MPI_Gatherv(tempMap[i].data(), localdata, Type, totalData, toRecvSize, disps, Type, i, Comm); // every PE sends the data that is saved for the destination rank.
        
        // append the arrays received to the vector to return
        if(currentRank == i) // only run this part when current rank == i
        {
            tempSave.insert(tempSave.end(), totalData, totalData + total);
            // delete varaibles used
            delete[] totalData;
            delete[] disps;
            delete[] toRecvSize;
        }
    }
    // assign the inout reference to tempSave
    inout = tempSave;
} // mpi_shuffle

#endif // A1_HPP