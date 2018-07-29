#include <map>
//console handling
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
//IO
#include <pcl/io/pcd_io.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <pcl/common/time.h>
//ICP
#include <pcl/registration/icp.h>
//EGI dependencies
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/transforms.h>
//dependencies of other registration techniques
//SCA-IA dependencies
#include <limits>
#include <vector>
#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/ia_ransac.h>
//PCA dependencies
#include <pcl/common/pca.h>
//#define M_PI (3.14159265358979323846)
#include <stdlib.h>

#ifndef _RGSTRTN_H_
#define _RGSTRTN_H_
//TRACE_CMH("BASE: [%d]\n", count++); 
//healpiX
class healpixSampling
{
public:
	healpixSampling(int n, pcl::PointCloud<pcl::PointXYZ> InputCloud);
	void generate();

	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
	Eigen::MatrixXf pts;
protected:
	int _n;
	pcl::PointCloud<pcl::PointXYZ> _inputcloud;
};

//map all normals onto pixels generated by healpiX sampling
class Ftr
{
public:
	Ftr(const healpixSampling &healpiX, const pcl::PointCloud<pcl::PointXYZ> &InputCloud, int Kneighbours = 1);
	void compute();
	void refresh(const Eigen::Matrix3f &M);
	void restore();
	Eigen::Matrix<float, Eigen::Dynamic, 3> find(int index, int Kpixels);

	Eigen::Matrix<int, Eigen::Dynamic, 1> density;
	std::map<int, std::vector<int>> normals_index;
	pcl::PointCloud<pcl::PointXYZ> pixelsCloud;
protected:
	healpixSampling _healpiX;
	pcl::PointCloud<pcl::PointXYZ> _InputCloud;
	pcl::PointCloud<pcl::PointXYZ> _InputCloud0;
	std::vector<int> intensity; std::vector<int> intensity0;
	Eigen::Matrix<int, Eigen::Dynamic, 1> density0;
	std::map<int, std::vector<int>> normals_index0;
	Eigen::MatrixXf normals0;
	int _Kneighbours;
	Eigen::MatrixXf normals;

	void backup();

};

//estimate rotation matrix
class RotationEstmtn
{
public:
	RotationEstmtn(const pcl::PointCloud<pcl::PointXYZ> &InputCloud1,
		const pcl::PointCloud<pcl::PointXYZ> &InputCloud2, pcl::PointCloud<pcl::PointXYZ> HEALPixcloud, int iterMax = 15, int stall = 3, int Kneighbours = 8,
		int steps = 10, int blocks = 20, float tolerance = 0.02);
	~RotationEstmtn();
	void compute();

	Eigen::Matrix<float, 3, 3> M;
	float fitness;
	int iter;
protected:
	Eigen::Matrix3f Rotationest();
	pcl::PointCloud<pcl::PointXYZ> _HEALPixcloud;
	pcl::PointCloud<pcl::PointXYZ> _InputCloud1;//model normals cloud
	pcl::PointCloud<pcl::PointXYZ> _InputCloud2;//model normals cloud
	int _stall, _Kneighbours, _steps, _blocks, _iterMax;
	float _tolerance;
	Ftr *ftr_model = NULL, *ftr_data = NULL;

	std::ofstream file;
	void mysort(Eigen::Matrix <float, 12, 5> &Matrix);
	class comparecol
	{
	public:
		bool operator()(const Eigen::Matrix <float, 1, 5> &a1, const Eigen::Matrix <float, 1, 5> &a2) const
		{
			return a1(0, 3) > a2(0, 3);
		}
	};
};

//2D search points lie within a r radius circle given its centerl point
class Sieve
{
public:
	Sieve(Eigen::Matrix<float, Eigen::Dynamic, 2> normals, float xmin, float xmax, float ymin, float ymax,
		float radius, int grids);
	void compute();

	Eigen::Matrix<float, 1, 2> maxpoint;
protected:
	float _xmin, _xmax, _ymin, _ymax;
	float _radius;
	int _grids;
	Eigen::VectorXf intensity;
	Eigen::Matrix<float, Eigen::Dynamic, 2> _normals;
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	pcl::PointCloud<pcl::PointXYZ>::Ptr normalsCloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>  cloud;
};

//storage results
class Results
{
public:
	Results(int samples, int argc);
	float RMS(const pcl::PointCloud<pcl::PointNormal> &inputcloud, const pcl::PointCloud<pcl::PointNormal> &targetcloud);
	void outputCSV(std::string filename);

	Eigen::MatrixXi iter;
	Eigen::MatrixXf fitness;
	Eigen::MatrixXf runtime;
	Eigen::MatrixXf rms;
	pcl::StopWatch watch;
	Eigen::MatrixXf angles;

protected:
	int _argc, _samples;
};

///////////////////////////////////////////////////class SAC-IA
class FeatureCloud
{
public:
	// A bit of shorthand
	typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
	typedef pcl::PointCloud<pcl::Normal> SurfaceNormals;
	typedef pcl::PointCloud<pcl::FPFHSignature33> LocalFeatures;
	typedef pcl::search::KdTree<pcl::PointXYZ> SearchMethod;

	FeatureCloud() :
		search_method_xyz_(new SearchMethod),
		normal_radius_(0.02f),
		feature_radius_(0.02f)
	{}

	~FeatureCloud() {}

	// Process the given cloud
	void
		setInputCloud(PointCloud::Ptr xyz)
	{
		xyz_ = xyz;
		processInput();
	}

	// Load and process the cloud in the given PCD file
	void
		loadInputCloud(const std::string &pcd_file)
	{
		xyz_ = PointCloud::Ptr(new PointCloud);
		pcl::io::loadPCDFile(pcd_file, *xyz_);
		processInput();
	}

	// Get a pointer to the cloud 3D points
	PointCloud::Ptr
		getPointCloud() const
	{
		return (xyz_);
	}

	// Get a pointer to the cloud of 3D surface normals
	SurfaceNormals::Ptr
		getSurfaceNormals() const
	{
		return (normals_);
	}

	// Get a pointer to the cloud of feature descriptors
	LocalFeatures::Ptr
		getLocalFeatures() const
	{
		return (features_);
	}

protected:
	// Compute the surface normals and local features
	void
		processInput()
	{
		computeSurfaceNormals();
		computeLocalFeatures();
	}

	// Compute the surface normals
	void
		computeSurfaceNormals()
	{
		normals_ = SurfaceNormals::Ptr(new SurfaceNormals);

		pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> norm_est;
		norm_est.setInputCloud(xyz_);
		norm_est.setSearchMethod(search_method_xyz_);
		norm_est.setRadiusSearch(normal_radius_);
		norm_est.compute(*normals_);
	}

	// Compute the local feature descriptors
	void
		computeLocalFeatures()
	{
		features_ = LocalFeatures::Ptr(new LocalFeatures);

		pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_est;
		fpfh_est.setInputCloud(xyz_);
		fpfh_est.setInputNormals(normals_);
		fpfh_est.setSearchMethod(search_method_xyz_);
		fpfh_est.setRadiusSearch(feature_radius_);
		fpfh_est.compute(*features_);
	}

private:
	// Point cloud data
	PointCloud::Ptr xyz_;
	SurfaceNormals::Ptr normals_;
	LocalFeatures::Ptr features_;
	SearchMethod::Ptr search_method_xyz_;

	// Parameters
	float normal_radius_;
	float feature_radius_;
};

class TemplateAlignment
{
public:

	// A struct for storing alignment results
	struct Result
	{
		float fitness_score;
		Eigen::Matrix4f final_transformation;
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};

	TemplateAlignment() :
		min_sample_distance_(0.05f),
		max_correspondence_distance_(0.01f*0.01f),
		nr_iterations_(500)
	{
		// Initialize the parameters in the Sample Consensus Initial Alignment (SAC-IA) algorithm
		sac_ia_.setMinSampleDistance(min_sample_distance_);
		sac_ia_.setMaxCorrespondenceDistance(max_correspondence_distance_);
		sac_ia_.setMaximumIterations(nr_iterations_);
	}

	~TemplateAlignment() {}

	// Set the given cloud as the target to which the templates will be aligned
	void
		setTargetCloud(FeatureCloud &target_cloud)
	{
		target_ = target_cloud;
		sac_ia_.setInputTarget(target_cloud.getPointCloud());
		sac_ia_.setTargetFeatures(target_cloud.getLocalFeatures());
	}

	// Add the given cloud to the list of template clouds
	void
		addTemplateCloud(FeatureCloud &template_cloud)
	{
		templates_ = template_cloud;
	}

	// Align the given template cloud to the target specified by setTargetCloud ()
	void
		align(FeatureCloud &template_cloud, TemplateAlignment::Result &result)
	{
		sac_ia_.setInputSource(template_cloud.getPointCloud());
		sac_ia_.setSourceFeatures(template_cloud.getLocalFeatures());

		pcl::PointCloud<pcl::PointXYZ> registration_output;
		sac_ia_.align(registration_output);

		result.fitness_score = (float)sac_ia_.getFitnessScore(max_correspondence_distance_);
		result.final_transformation = sac_ia_.getFinalTransformation();
	}

private:
	// A list of template clouds and the target to which they will be aligned
	FeatureCloud templates_;
	FeatureCloud target_;

	// The Sample Consensus Initial Alignment (SAC-IA) registration routine and its parameters
	pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia_;
	float min_sample_distance_;
	float max_correspondence_distance_;
	int nr_iterations_;
};

class Sac_ia
{
public:
	Sac_ia(pcl::PointCloud<pcl::PointXYZ>::Ptr inputcloud, pcl::PointCloud<pcl::PointXYZ>::Ptr targetcloud)
	{
		_inputcloud = inputcloud;
		_targetcloud = targetcloud;
		// ...removing distant points
		const float depth_limit = 1.0;
		pass.setInputCloud(_inputcloud);
		pass.setFilterFieldName("z");
		pass.setFilterLimits(0, depth_limit);
		pass.filter(*_inputcloud);
		// ... and downsampling the point cloud
		const float voxel_grid_size = 0.005f;
		pcl::VoxelGrid<pcl::PointXYZ> vox_grid;
		vox_grid.setInputCloud(_inputcloud);
		vox_grid.setLeafSize(voxel_grid_size, voxel_grid_size, voxel_grid_size);
		pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud(new pcl::PointCloud<pcl::PointXYZ>);
		vox_grid.filter(*tempCloud);
		_inputcloud = tempCloud;

		// Assign to the target FeatureCloud
		FeatureCloud source_cloud;
		FeatureCloud target_cloud;
		target_cloud.setInputCloud(_targetcloud);
		source_cloud.setInputCloud(_inputcloud);
		// Set the TemplateAlignment inputs
		TemplateAlignment template_align;
		// Assign to the template FeatureCloud
		template_align.setTargetCloud(source_cloud);

		template_align.align(target_cloud, Results);
		
	}
	~Sac_ia()
	{
	}

	TemplateAlignment::Result Results;
private:
	pcl::PointCloud<pcl::PointXYZ>::Ptr _inputcloud;
	pcl::PointCloud<pcl::PointXYZ>::Ptr _targetcloud;
	pcl::PassThrough<pcl::PointXYZ> pass;
	pcl::VoxelGrid<pcl::PointXYZ> vox_grid;
};

class Pcaalign
{
public:
	Pcaalign(pcl::PointCloud<pcl::PointXYZ>::Ptr inputcloud, pcl::PointCloud<pcl::PointXYZ>::Ptr targetcloud):
		_inputcloud(inputcloud), _targetcloud(targetcloud)
	{
	}
	void compute()
	{
		M = Eigen::Matrix4f::Identity();
		//initialize pca object
		pcainput.setInputCloud(_inputcloud);
		pcatarget.setInputCloud(_targetcloud);

		//get eigen vectors of two point cloud
		Eigen::Matrix3f eigeninput, eigentarget;
		eigeninput = pcainput.getEigenVectors();
		eigentarget = pcatarget.getEigenVectors();
		
		//compute rotation matrix
		M.block(0, 0, 3, 3) = eigeninput * eigentarget.inverse();
		pcl::transformPointCloud(*_targetcloud, *tempcloud, M);
		Eigen::Vector3f inputpoint, targetpoint;
		targetpoint << tempcloud->points[1].x, tempcloud->points[1].y, tempcloud->points[1].z;
		inputpoint << _inputcloud->points[1].x, _inputcloud->points[1].y, _inputcloud->points[1].z;
		for (size_t i =	0; i < 3; i++)
		{
			if(targetpoint[i] * inputpoint[i] < 0)
			{
				M.block(0, 0, 3, 3).row(i) = -M.block(0, 0, 3, 3).row(i);
			}
		}
	}
	~Pcaalign();
	
	Eigen::Matrix4f M;
private:
	pcl::PointCloud<pcl::PointXYZ>::Ptr _inputcloud;
	pcl::PointCloud<pcl::PointXYZ>::Ptr _targetcloud;
	pcl::PointCloud<pcl::PointXYZ>::Ptr tempcloud;
	pcl::PCA<pcl::PointXYZ> pcainput;
	pcl::PCA<pcl::PointXYZ> pcatarget;
};
#endif