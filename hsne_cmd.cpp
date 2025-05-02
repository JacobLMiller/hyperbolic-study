#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "hdi/dimensionality_reduction/hierarchical_sne.h"
#include "hdi/dimensionality_reduction/hierarchical_sne_inl.h"
#include "hdi/dimensionality_reduction/sparse_tsne_user_def_probabilities.h"
#include "hdi/dimensionality_reduction/tsne_parameters.h"

#include "hdi/utils/math_utils.h"
#include "hdi/utils/log_helper_functions.h"
#include "hdi/utils/scoped_timers.h"
#include "hdi/utils/memory_utils.h"
#include "hdi/utils/log_progress.h"
#include "hdi/utils/graph_algorithms.h"

#include "hdi/data/map_mem_eff.h"
#include "hdi/data/map_helpers.h"
#include "hdi/data/io.h"

#include <vector>
#include <stdint.h>
#include <cmath>
#include <cassert>
#include <cmath>
#include <math.h> 
#include <iostream>

namespace py = pybind11;
using namespace hdi::dr; 


typedef std::vector<hdi::data::MapMemEff<uint32_t, float>> sparse_scalar_matrix_type;
typedef float scalar_type;
typedef HierarchicalSNE<scalar_type, sparse_scalar_matrix_type> HSNE;
typedef HSNE::Scale scale_type;

typedef std::tuple<unsigned int, unsigned int> id_type;
typedef hdi::dr::SparseTSNEUserDefProbabilities<scalar_type,sparse_scalar_matrix_type> tsne_type;
typedef hdi::data::Embedding<scalar_type> embedding_type;


class PyHierarchicalSNE {
public:
    PyHierarchicalSNE() : hsne_() {}

    py::tuple sparse_matrix_to_csr(sparse_scalar_matrix_type &matrix){
        std::vector<int> row_indices, col_indices;
        std::vector<float> values;

        int row_idx = 0;
        for (const auto &row : matrix){
            for (const auto &[col, value] : row){
                row_indices.push_back(row_idx);
                col_indices.push_back(col);
                values.push_back(value);
            }
            row_idx ++;
        }

        return py::make_tuple(row_indices, col_indices, values);
    }

    void computeEmbedding(sparse_scalar_matrix_type P) {
        hdi::dr::TsneParameters params = hdi::dr::TsneParameters();
        double theta = (P.size() < 1000) ? 0.0 : (P.size() < 15000) ? (P.size() - 1000.) / (15000. - 1000.) * 0.5 : 0.5;
        params._exaggeration_factor = (P.size() < 1000) ? 1.5 : (P.size() < 15000) ? 1.5 + (P.size() - 1000.) / (15000. - 1000.) * 8.5 : 10;
        params._remove_exaggeration_iter = 170;

        tsne_.setTheta(theta);
        tsne_.initialize(P, &embedding_, params);

        while (tsne_.iteration() < 1500) {
            tsne_.doAnIteration();
        }
    }

    void computeEmbeddingX(sparse_scalar_matrix_type P, embedding_type& X) {
        hdi::dr::TsneParameters params = hdi::dr::TsneParameters();

        // Adjust theta and exaggeration factor based on dataset size
        double theta = (P.size() < 1000) ? 0.0 : (P.size() < 15000) ? 
                    (P.size() - 1000.) / (15000. - 1000.) * 0.5 : 0.5;
        params._exaggeration_factor = (P.size() < 1000) ? 1.5 : (P.size() < 15000) ? 
                                    1.5 + (P.size() - 1000.) / (15000. - 1000.) * 8.5 : 10;
        params._remove_exaggeration_iter = 170;

        tsne_.setTheta(theta);

        tsne_.initialize(P, &X, params);

        // Perform t-SNE iterations
        while (tsne_.iteration() < 1500) {
            tsne_.doAnIteration();
        }
    }


    py::array getEmbeddingAtScale(unsigned int scale){
        scale_type scaleObj = hsne_.scale(scale);
        computeEmbedding(scaleObj._transition_matrix);
        return getEmbedding();
    }

    py::array getEmbedding() {
        return py::array(embedding_.getContainer().size(), embedding_.getContainer().data());
    }

    py::array getIdxAtScale(unsigned int scale){
        scale_type scaleObj = hsne_.scale(scale);
        std::vector<uint32_t> idx = scaleObj._landmark_to_original_data_idx;
        return py::array(idx.size(), idx.data());
    }

    std::map<unsigned int, scalar_type> getLandmarks(unsigned int scale, py::array_t<int> selection, std::vector<unsigned int> &selection_idxes) {
        
        py::buffer_info buf = selection.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("Expected a 1D NumPy array");
        }

        int* ptrdata = static_cast<int*>(buf.ptr);
        selection_idxes.assign(ptrdata, ptrdata + buf.shape[0]);

        // std::cout << "got selectionidxes" << std::endl;


        std::map<unsigned int, scalar_type> neighbors;
        hsne_.getInfluencedLandmarksInPreviousScale(scale, selection_idxes, neighbors);

        return neighbors;
    }

    std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, std::vector<float>,std::vector<unsigned int>>
     drillDownMatrix(unsigned int scale, const py::array_t<int>& selection){
        std::vector<unsigned int> selection_idx;

        // std::cout << "entered function" << std::endl;
        // std::cout << scale << std::endl;

        std::map<unsigned int, scalar_type> landmarks = getLandmarks(scale, selection, selection_idx);

        // std::cout << "got landmarks" << std::endl;

        scalar_type gamma = 0.5;

        std::vector<unsigned int> landmarks_to_add;
        for (auto & n : landmarks){
            if (n.second > gamma){
                landmarks_to_add.push_back(n.first);
            }
        }

        // std::cout << "Assigned set" << std::endl;

        sparse_scalar_matrix_type new_matrix;
        std::vector<unsigned int> new_idxes;        

        std::vector<uint32_t> row_indices, col_indices; 
        std::vector<float> values;        

        // std::cerr << "Scale: " << scale << std::endl;
        // std::cerr << "Transition matrix size: " << hsne_.scale(scale)._transition_matrix.size() << std::endl;
        // std::cerr << "Landmarks to add size: " << landmarks_to_add.size() << std::endl;
        // std::cerr << "Selection index size: " << selection_idx.size() << std::endl;

        if (hsne_.scale(scale)._transition_matrix.size() == 0) {
            std::cerr << "Error: Transition matrix is empty!" << std::endl;
        }

        if (landmarks_to_add.empty()) {
            std::cerr << "Error: No landmarks provided!" << std::endl;
            return {row_indices, col_indices, values, new_idxes};
        }

        if (selection_idx.empty()) {
            std::cerr << "Warning: selection_idx is empty before extractSubGraph!" << std::endl;
        }

        hdi::utils::extractSubGraph(
            hsne_.scale(scale-1)._transition_matrix,
            landmarks_to_add,
            new_matrix,
            new_idxes, 
            1
        );



        for (uint32_t row=0; row<new_matrix.size(); ++row){
            for(const auto& [col,val] : new_matrix[row]){
                row_indices.push_back(row);
                col_indices.push_back(col);
                values.push_back(val);
            }
        }

      

        return {row_indices, col_indices, values, new_idxes};

    }    

    void addScale() {
        hsne_.addScale();
    }

    std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, std::vector<float>>
    getMatrixAtTopScale(){

        scale_type scale0 = hsne_.top_scale();
        sparse_scalar_matrix_type transition = scale0._transition_matrix;
        

        std::vector<uint32_t> row_indices, col_indices; 
        std::vector<float> values;                

        for (uint32_t row=0; row<transition.size(); ++row){
            for(const auto& [col,val] : transition[row]){
                row_indices.push_back(row);
                col_indices.push_back(col);
                values.push_back(val);
            }
        }

        return {row_indices, col_indices, values};

    }

    static std::shared_ptr<PyHierarchicalSNE> initialize(int num_points, int num_dim, int num_scales, py::array_t<float> data) {

        // std::cout << "Hello world" << std::endl;

        auto hsne_instance = std::make_shared<PyHierarchicalSNE>();

        py::buffer_info buf = data.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("Expected a 1D numpy array");
        }

        float* ptrdata = static_cast<float*>(buf.ptr);

        hsne_instance->hsne_.setDimensionality(num_dim);
        hsne_instance->hsne_.initialize(ptrdata, num_points, hsne_instance->params_);

        for (int i = 0; i < num_scales - 1; ++i) {
            hsne_instance->hsne_.addScale();
        }

        // scale_type scale0 = hsne_instance->hsne_.top_scale();
        // sparse_scalar_matrix_type transition = scale0._transition_matrix;

        // hsne_instance->computeEmbedding(transition);

        return hsne_instance;
    }

private:
    HSNE hsne_;
    HSNE::Parameters params_;
    tsne_type tsne_;
    embedding_type embedding_;
};

// Expose class to Python
PYBIND11_MODULE(hsne_wrapper, m) {
    py::class_<PyHierarchicalSNE, std::shared_ptr<PyHierarchicalSNE>>(m, "PyHierarchicalSNE")
        .def(py::init<>())  // Default constructor
        .def("computeEmbedding", &PyHierarchicalSNE::computeEmbedding)
        .def("getEmbedding", &PyHierarchicalSNE::getEmbedding)
        .def("addScale", &PyHierarchicalSNE::addScale)
        .def("getEmbeddingAtScale", &PyHierarchicalSNE::getEmbeddingAtScale)
        .def("getIdxAtScale", &PyHierarchicalSNE::getIdxAtScale)
        .def("drillDownMatrix", &PyHierarchicalSNE::drillDownMatrix)
        .def("getMatrixAtTopScale", &PyHierarchicalSNE::getMatrixAtTopScale)
        .def_static("initialize", &PyHierarchicalSNE::initialize);  // Static method to create an instance
}