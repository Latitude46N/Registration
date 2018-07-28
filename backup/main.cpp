#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <Eigen/Core>
#include <stdlib.h>
#include <pcl/common/time.h>

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
//healpixSampling methods
healpixSampling::healpixSampling(int n, pcl::PointCloud<pcl::PointXYZ> InputCloud) : _n(n), _inputcloud(InputCloud)
{
};

void
healpixSampling::generate()
{
	*cloud = _inputcloud;
	//build ketree ready to search
	kdtree.setInputCloud(cloud);
	pts = cloud->getMatrixXfMap().adjoint().leftCols(3);
}


class Ftr
{
public:
	Ftr(const healpixSampling &healpiX, const pcl::PointCloud<pcl::PointXYZ> &InputCloud, int Kneighbours = 1);
	void compute();
	void refresh(const Eigen::Matrix3f &M);
	void restore();
	Eigen::Matrix<float, Eigen::Dynamic, 3> find(pcl::PointXYZ point, int Kpixels);

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
//Ftr methods
Ftr::Ftr(const healpixSampling &healpiX, const pcl::PointCloud<pcl::PointXYZ> &InputCloud, int Kneighbours)
	:_healpiX(healpiX), _InputCloud(InputCloud), _Kneighbours(Kneighbours)
{
}

void
Ftr::refresh(const Eigen::Matrix3f &M)
{
	backup();
	
	Eigen::Matrix4f T = Eigen::Matrix4f::Identity();//
	pcl::PointCloud<pcl::PointXYZ> _InputCloudtmp;
	T.block(0, 0, 3, 3) = M;
	pcl::transformPointCloud(_InputCloud, _InputCloudtmp, T);
	_InputCloud = _InputCloudtmp;
	
	std::vector<int> pointIdxNKNSearch;
	std::vector<float> k_sqr_distances;

	intensity.resize(_healpiX.pts.rows());
	
	for (int i = 0; i < _InputCloud.points.size(); ++i)
	{
		_healpiX.kdtree.nearestKSearch(_InputCloud.points[i], _Kneighbours, pointIdxNKNSearch, k_sqr_distances);
		for (size_t j = 0; j < _Kneighbours; j++)
		{
			intensity[pointIdxNKNSearch[j]] ++;
			normals_index[pointIdxNKNSearch[j]].insert(normals_index[pointIdxNKNSearch[j]].end(), i);
		}
	}

	normals = _InputCloud.getMatrixXfMap().adjoint().leftCols(3);
	Eigen::VectorXi density_tmp = Eigen::Map<Eigen::VectorXi>(intensity.data(), intensity.size());
	density = density_tmp;
}

void
Ftr::backup()
{
	intensity0.clear();
	normals_index0.clear();
	_InputCloud0 = _InputCloud;
	intensity0 = intensity;
	density0 = density;
	normals_index0 = normals_index;
	normals0 = normals;
	intensity.clear();
	normals_index.clear();
}

void
Ftr::restore()
{
	intensity.clear();
	normals_index.clear();
	_InputCloud = _InputCloud0;
	intensity = intensity0;
	density = density0;
	normals_index = normals_index0;
	normals = normals0;
	intensity0.clear();
	normals_index0.clear();
}

Eigen::Matrix<float, Eigen::Dynamic, 3>
Ftr::find(pcl::PointXYZ point, int Kpixels)
{
	Eigen::Matrix<float, Eigen::Dynamic, 3> nearestnormals;
	std::vector<int> pointIdxNKNSearch;
	std::vector<int> normalsmapIdx;
	std::vector<int> normalsIdx;
	std::vector<float> k_sqr_distances;
	_healpiX.kdtree.nearestKSearch(point, Kpixels, pointIdxNKNSearch, k_sqr_distances);

	for (size_t i = 0; i < pointIdxNKNSearch.size(); i++)
	{
		normalsmapIdx.clear();
		normalsmapIdx = normals_index[pointIdxNKNSearch[i]];
		normalsIdx.insert(normalsIdx.end(), std::begin(normalsmapIdx), std::end(normalsmapIdx));
	}

	nearestnormals.conservativeResize(normalsIdx.size(), 3);
	for (size_t i = 0; i < normalsIdx.size(); i++)
	{
		nearestnormals.row(i) = normals.row(normalsIdx[i]);
	}
	return nearestnormals;
}

void
Ftr::compute()
{
	std::vector<int> pointIdxNKNSearch(_Kneighbours);
	std::vector<float> k_sqr_distances(_Kneighbours);

	intensity.resize(_healpiX.pts.rows());
	
	for (int i = 0; i < _InputCloud.points.size(); ++i)
	{
		_healpiX.kdtree.nearestKSearch(_InputCloud.points[i], _Kneighbours, pointIdxNKNSearch, k_sqr_distances);
		for (size_t j = 0; j < _Kneighbours; j++)
		{
			intensity[pointIdxNKNSearch[j]] ++;
			normals_index[pointIdxNKNSearch[j]].insert(normals_index[pointIdxNKNSearch[j]].end(), i);
		}
	}

	normals = _InputCloud.getMatrixXfMap().adjoint().leftCols(3);
	pixelsCloud = *_healpiX.cloud;
	Eigen::VectorXi density_tmp = Eigen::Map<Eigen::VectorXi>(intensity.data(), intensity.size());
	density = density_tmp;
}

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
//class Sieve methods
Sieve::Sieve(Eigen::Matrix<float, Eigen::Dynamic, 2> normals, float xmin, float xmax, float ymin, float ymax, float radius, int grids) :
	_normals(normals), _xmin(xmin), _xmax(xmax), _ymin(ymin), _ymax(ymax), _radius(radius), _grids(grids) {};

void
Sieve::compute()
{
	int size = pow(_grids, 2);
	cloud.width = size;
	cloud.height = 1;
	cloud.points.resize(cloud.width * cloud.height);
	for (size_t i = 0; i < _grids; ++i)
	{
		for (size_t j = 0; j < _grids; ++j)
		{
			cloud.points[i*_grids + j].x = (_xmax - _xmin) / (_grids - 1) * i + _xmin;
			cloud.points[i*_grids + j].y = (_ymax - _ymin) / (_grids - 1) * j + _ymin;
			cloud.points[i*_grids + j].z = 0;

		}
	}
	normalsCloud->width = _normals.rows();
	normalsCloud->height = 1;
	normalsCloud->points.resize(normalsCloud->width * normalsCloud->height);
	for (int i = 0; i < _normals.rows(); ++i)
	{
		normalsCloud->points[i].x = _normals(i, 0);
		normalsCloud->points[i].y = _normals(i, 1);
		normalsCloud->points[i].z = 0;
	}
	
	kdtree.setInputCloud(normalsCloud);
	std::vector<int> pointIdxRadiusSearch;
	std::vector<float> k_sqr_distances;
	intensity.conservativeResize(size, 1);
	for (int i = 0; i < size; ++i)
	{
		kdtree.radiusSearch(cloud.points[i], _radius, pointIdxRadiusSearch, k_sqr_distances);
		intensity[i] = (int)pointIdxRadiusSearch.size();
	}

	int max_index;
	intensity.maxCoeff(&max_index);
	maxpoint << cloud.points[max_index].x, cloud.points[max_index].y;
}
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
//class Results
Results::Results(int samples, int argc)
{

	_argc = argc;
	_samples = samples;
	iter = Eigen::MatrixXi(_samples, _argc - 1);
	fitness = Eigen::MatrixXf(_samples, _argc - 1);
	runtime = Eigen::MatrixXf(_samples, _argc - 1);
	rms = Eigen::MatrixXf(_samples, _argc - 1);
	angles = Eigen::MatrixXf(_samples, 3);
}

float
Results::RMS(const pcl::PointCloud<pcl::PointNormal> &inputcloud, const pcl::PointCloud<pcl::PointNormal> &targetcloud)
{
	float RMS = 0;
	Eigen::MatrixXf inputcloudM = inputcloud.getMatrixXfMap().adjoint().leftCols(3);
	Eigen::MatrixXf targetcloudM = targetcloud.getMatrixXfMap().adjoint().leftCols(3);
	for (int i = 0; i < 3; i++)
	{
		inputcloudM.col(i) = inputcloudM.col(i) / inputcloudM.col(i).maxCoeff();
		targetcloudM.col(i) = targetcloudM.col(i) / targetcloudM.col(i).maxCoeff();
	}
	for (size_t i = 0; i < inputcloudM.rows(); ++i)
	{
		RMS = RMS + pow(inputcloudM.row(i).dot(targetcloudM.row(i)), 2);
	}
	RMS = sqrt(RMS / (float)inputcloud.points.size());
	return RMS;
}

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
//RotationEstmtn methods
RotationEstmtn::RotationEstmtn(const pcl::PointCloud<pcl::PointXYZ> &InputCloud1,
	const pcl::PointCloud<pcl::PointXYZ> &InputCloud2, pcl::PointCloud<pcl::PointXYZ> HEALPixcloud, int iterMax, int stall, int Kneighbours,
	int steps, int blocks, float tolerance)
	: _InputCloud1(InputCloud1), _InputCloud2(InputCloud2), _HEALPixcloud(HEALPixcloud), _iterMax(iterMax), _stall(stall),
	_Kneighbours(Kneighbours), _steps(steps), _blocks(blocks), _tolerance(tolerance)
{
}

RotationEstmtn::~RotationEstmtn()
{
	delete ftr_model;
	delete ftr_data;
}

void
RotationEstmtn::compute()
{
	//output results
	file.open("G:/PROJECT/test1/build/Debug/test.txt");
	//
	Eigen::Vector3f rand;
	Eigen::Array<float, 3, 1> rand_sin, rand_cos;
	Eigen::Matrix3f T_rot, T_tmp;
	rand = Eigen::Vector3f::Random() * M_PI;
	rand_sin = rand.array().sin();
	rand_cos = rand.array().cos();
	T_rot = (Eigen::MatrixXf(3, 3) << 1, 0, 0, 0, rand_cos(0, 0), rand_sin(0, 0), 0, -rand_sin(0, 0), rand_cos(0, 0)).finished() *
		(Eigen::MatrixXf(3, 3) << rand_cos(1, 0), 0, -rand_sin(1, 0), 0, 1, 0, rand_sin(1, 0), 0, rand_cos(1, 0)).finished() *
		(Eigen::MatrixXf(3, 3) << rand_cos(2, 0), rand_sin(2, 0), 0, -rand_sin(2, 0), rand_cos(2, 0), 0, 0, 0, 1).finished();

	M.setIdentity();

	//
	file << "rand sampling of angles\n" << rand << '\n';/////////////////
	file << "rand_sin\n" << rand_sin << '\n';/////////////////
	file << "rand_cos\n" << rand_cos << '\n';/////////////////
	file << "T_rot\n" << T_rot << '\n';/////////////////

	healpixSampling healpiX(16, _HEALPixcloud);
	healpiX.generate();

	ftr_model = new Ftr(healpiX, _InputCloud1);
	ftr_model->compute();
	ftr_data = new Ftr(healpiX, _InputCloud2);
	ftr_data->compute();
	//rotation
	iter = 0;
	int stall = 0;
	ftr_data;//copying a object, extra attention
	float fitness_tmp;
	fitness = 0;
	//iteration
	std::ofstream file_density("G:/PROJECT/test1/build/Debug/densitytest1.txt");
	while (fitness < 1 - _tolerance && iter < _iterMax)
	{
		file <<"//////////////////////////////////////////" << '\n';/////////////////
		file << "iteration\n" << iter+1 << '\n';/////////////////
		T_tmp = Rotationest();
		ftr_data->refresh(T_tmp);
		fitness_tmp = (float)ftr_model->density.dot(ftr_data->density) / (float)(ftr_model->density.norm() * ftr_data->density.norm());

		file_density << "iteration" << iter + 1 << std::endl;
		file_density << ftr_model->density.adjoint() << std::endl;
		file_density << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
		file_density << ftr_data->density.adjoint() << std::endl;

		stall++;
		if (fitness < fitness_tmp)
		{
			if (fitness_tmp > fitness * 1.005) stall--;
			fitness = fitness_tmp;
			M = T_tmp * M;
		}
		else
		{
			ftr_data->restore();
		}

		if (stall >= 3 && fitness_tmp < 1 - _tolerance)
		{
			M = T_rot * M;
			ftr_data->refresh(T_rot);
			stall = 0;
		}

		iter++;
		//
		file << "M\n" << M << '\n';/////////////////
		file << "stall\n" << stall << '\n';/////////////////
		file << "fitness\n" << fitness << '\n';/////////////////
	}
}

Eigen::Matrix3f
RotationEstmtn::Rotationest()
{
	std::ofstream file1("G:/PROJECT/test1/build/Debug/test1.txt", std::ios::out|std::ios::app);

	Eigen::Matrix3f T;
	Eigen::Matrix <int, 12, 2> ind_md, ind_dd;
	Eigen::Matrix <float, 12, 5> pts_md, pts_dd;
	int pixel_size = ftr_model->density.size() / 12;

	for (int i = 0; i < 12; ++i)
	{
		ind_md(i, 0) = ftr_model->density.segment(pixel_size*i, pixel_size).maxCoeff(&ind_md(i, 1));
		ind_md(i, 1) = ind_md(i, 1) + pixel_size * i;
		ind_dd(i, 0) = ftr_data->density.segment(pixel_size*i, pixel_size).maxCoeff(&ind_dd(i, 1));
		ind_dd(i, 1) = ind_dd(i, 1) + pixel_size * i;
		
		pts_md.row(i) << ftr_model->pixelsCloud.points[ind_md(i, 1)].x, ftr_model->pixelsCloud.points[ind_md(i, 1)].y, ftr_model->pixelsCloud.points[ind_md(i, 1)].z, ind_md(i, 0), ind_md(i, 1);
		pts_dd.row(i) << ftr_data->pixelsCloud.points[ind_dd(i, 1)].x, ftr_data->pixelsCloud.points[ind_dd(i, 1)].y, ftr_data->pixelsCloud.points[ind_dd(i, 1)].z, ind_dd(i, 0), ind_dd(i, 1);
	}
	//sort
	mysort(pts_md);
	mysort(pts_dd);

	file << "	" << "pts_md before selection\n" << pts_md.topRows(2) << '\n';/////////////////
	file << "	" << "pts_dd before selection\n" << pts_dd.topRows(2) << '\n';/////////////////
	
	//
	int i = 1;
	while (abs(pts_md.row(0).head(3).dot(pts_md.row(i).head(3))) > 0.94)
		i++;
	pts_md.row(1) = pts_md.row(i);

	//
	for (i = 1; i < 12; ++i)
	{
		if (abs((pts_dd.row(i).head(3) - pts_dd.row(0).head(3)).norm()
			- (pts_md.row(1).head(3) - pts_md.row(0).head(3)).norm()) < 0.1)
		{
			pts_dd.row(1) = pts_dd.row(i);
			break;
		}
	}
	file << "	" << "pts_md after selection\n" << pts_md.topRows(2) << '\n';/////////////////
	file << "	" << "pts_dd after selection\n" << pts_dd.topRows(2) << '\n';/////////////////

	//
	Eigen::MatrixXf normalsXY_md, normalsXY_dd;
	Sieve *sieve_md = NULL, *sieve_dd = NULL;
	float xi_md, xi_dd;
	Eigen::Vector3f x_md, y_md, x_dd, y_dd, z_md, z_dd;;
	for (int i = 0; i < 2; ++i)
	{
		z_md << pts_md.row(i)[0], pts_md.row(i)[1], pts_md.row(i)[2];
		z_dd << pts_dd.row(i)[0], pts_dd.row(i)[1], pts_dd.row(i)[2];
		xi_md = -0.5 / pts_md(i, 2)*(pts_md(i, 0) + pts_md(i, 1));
		xi_dd = -0.5 / pts_dd(i, 2)*(pts_dd(i, 0) + pts_dd(i, 1));
		x_md = ((Eigen::MatrixXf(3, 1) << 0.5, 0.5, xi_md).finished() / (Eigen::MatrixXf(3, 1) << 0.5, 0.5, xi_md).finished().norm());
		x_dd = ((Eigen::MatrixXf(3, 1) << 0.5, 0.5, xi_dd).finished() / (Eigen::MatrixXf(3, 1) << 0.5, 0.5, xi_dd).finished().norm());
		y_md = z_md.cross(x_md);
		y_dd = z_dd.cross(x_dd);

		//
		file1 << "point selected pts_md:/n" << pts_md.row(i).head(3) << std::endl;
		file1 << "point selected pts_dd:/n" << pts_dd.row(i).head(3) << std::endl;
		file << "	" << "X-axis model:\n" << x_md << '\n';/////////////////
		file << "	" << "Y-axis model:\n" << y_md << '\n';/////////////////
		file << "	" << "Z-axis model:\n" << pts_md.row(i) << '\n';/////////////////
		file << "	" << "X-axis data:\n" << x_dd << '\n';/////////////////
		file << "	" << "Y-axis data:\n" << y_dd << '\n';/////////////////
		file << "	" << "Z-axis data:\n" << pts_dd.row(i) << '\n';/////////////////

		//project to a plane perpendicular to pts_md.row(i).head(3)
		normalsXY_md.resize(0, 0);
		normalsXY_dd.resize(0, 0);
		normalsXY_md = ((Eigen::MatrixXf(2, 3) << x_md.adjoint(), y_md.adjoint()).finished() * ftr_model->find(pcl::PointXYZ(pts_md(i, 0), pts_md(i, 1), pts_md(i, 2)), _Kneighbours).adjoint()).adjoint();
		normalsXY_dd = ((Eigen::MatrixXf(2, 3) << x_dd.adjoint(), y_dd.adjoint()).finished() * ftr_data->find(pcl::PointXYZ(pts_dd(i, 0), pts_dd(i, 1), pts_dd(i, 2)), _Kneighbours).adjoint()).adjoint();

		file1 << "normals near selected points models:/n" << ftr_model->find(pcl::PointXYZ(pts_md(i, 0), pts_md(i, 1), pts_md(i, 2)), _Kneighbours) << std::endl;
		file1 << "normals near selected points data:/n" << ftr_data->find(pcl::PointXYZ(pts_dd(i, 0), pts_dd(i, 1), pts_dd(i, 2)), _Kneighbours) << std::endl;

		file << "	" << "average of normalsXY_md:\n" << normalsXY_md.col(0).mean() << " " << normalsXY_md.col(1).mean() << '\n';/////////////////
		file << "	" << "average of normalsXY_dd:\n" << normalsXY_dd.col(0).mean() << " " << normalsXY_dd.col(1).mean() << '\n';/////////////////

		//
		sieve_md = new Sieve(normalsXY_md, normalsXY_md.col(0).minCoeff(), normalsXY_md.col(0).maxCoeff(), normalsXY_md.col(1).minCoeff(),
			normalsXY_md.col(1).maxCoeff(), (normalsXY_md.col(0).maxCoeff() - normalsXY_md.col(0).minCoeff()) / _steps, _blocks);
		sieve_dd = new Sieve(normalsXY_dd, normalsXY_dd.col(0).minCoeff(), normalsXY_dd.col(0).maxCoeff(), normalsXY_dd.col(1).minCoeff(),
			normalsXY_dd.col(1).maxCoeff(), (normalsXY_dd.col(0).maxCoeff() - normalsXY_dd.col(0).minCoeff()) / _steps, _blocks);
		sieve_md->compute();
		sieve_dd->compute();

		file << "	" << "average of normalsXY_md:\n" << normalsXY_md.col(0).mean() << " " << normalsXY_md.col(1).mean() << '\n';/////////////////
		file << "	" << "average of normalsXY_dd:\n" << normalsXY_dd.col(0).mean() << " " << normalsXY_dd.col(1).mean() << '\n';/////////////////
		
		pts_md.row(i).head(3) = sieve_md->maxpoint[0] * x_md.adjoint() + sieve_md->maxpoint[1] * y_md.adjoint() + sqrt(1 - pow(sieve_md->maxpoint[0], 2) - pow(sieve_md->maxpoint[1], 2)) * pts_md.row(i).head(3);
		pts_dd.row(i).head(3) = sieve_dd->maxpoint[0] * x_dd.adjoint() + sieve_dd->maxpoint[1] * y_dd.adjoint() + sqrt(1 - pow(sieve_dd->maxpoint[0], 2) - pow(sieve_dd->maxpoint[1], 2)) * pts_dd.row(i).head(3);

		file << "	" << "sieve_md->maxpoint:\n" << sieve_md->maxpoint << '\n';/////////////////
		file << "	" << "sieve_dd->maxpoint:\n" << sieve_dd->maxpoint << '\n';/////////////////

		delete sieve_dd;
		delete sieve_md;
	}
	
	file << "	" << "pts_md after repixelization\n" << pts_md.topRows(2) << '\n';/////////////////
	file << "	" << "pts_dd after repixelization\n" << pts_dd.topRows(2) << '\n';/////////////////

	//rotation vectors construction
	Eigen::Vector3f tmp_md, tmp_dd;
	tmp_md << pts_md.row(0)[0], pts_md.row(0)[1], pts_md.row(0)[2];
	tmp_dd << pts_dd.row(0)[0], pts_dd.row(0)[1], pts_dd.row(0)[2];
	Eigen::Vector3f r1 = (tmp_dd.cross(tmp_md) / tmp_dd.cross(tmp_md).norm()).adjoint();
	float theta1 = acos(pts_dd.row(0).head(3).dot(pts_md.row(0).head(3)));
	Eigen::AngleAxisf R1(theta1, r1);

	Eigen::Vector3f r2;
	r2 << pts_md.row(0)[0], pts_md.row(0)[1], pts_md.row(0)[2];
	Eigen::Vector3f a = pts_md.row(1).head(3).adjoint() - pts_md.row(1).head(3).dot(r2) * r2;
	Eigen::Vector3f b = R1.matrix() * pts_dd.row(1).head(3).adjoint() - (R1.matrix() * pts_dd.row(1).head(3).adjoint()).dot(r2) * r2;
	float theta2 = acos(a.dot(b)) / (a.norm() * b.norm());
	Eigen::AngleAxisf R2(theta2, r2);

	file << "	" << "r1 theta1 \n" << r1 << theta1 << '\n';/////////////////
	file << "	" << "r2 theta2 \n" << r2 << theta2 << '\n';/////////////////

	T = R2.matrix() * R1.matrix();

	file << "	" << "estimated rotation matrix \n" << T << '\n';/////////////////

	return T;

}

void
RotationEstmtn::mysort(Eigen::Matrix <float, 12, 5> &Matrix)
{
	std::vector<Eigen::Matrix <float, 1, 5>> sortmat;
	for (size_t i = 0; i < Matrix.rows(); i++)
	{
		sortmat.push_back(Matrix.row(i));
	}
	std::sort(sortmat.begin(), sortmat.end(), comparecol());
	for (size_t i = 0; i < Matrix.rows(); i++)
	{
		Matrix.row(i) = sortmat[i];
	}
}
void
Results::outputCSV(std::string filename)
{
	//file stream  
	std::ofstream oFile;
	std::string extension = ".csv";

	//open file after discard all its contents 
	oFile.open(filename + extension, std::ofstream::out | std::ofstream::trunc);
	oFile << "No." << "," << "Rotation Angles(X)" << "," << "Rotation Angles(Y)" << "," << "Rotation Angles(Z)" << std::endl;
	for (size_t i = 0; i < _samples; i++)
	{
		oFile << i + 1 << "," << angles.row(i)[0] << "," << angles.row(i)[1] << "," << angles.row(i)[2] << std::endl;
	}
	for (size_t i = 0; i < _argc-1; i++)
	{
		oFile << "The" << i + 1 << "-th point cloud" << std::endl;
		oFile << "No." << "," << "Run Time(ms)" << "," << "RMS" << "," << "Corr" << "," << "Iterations" << std::endl;
		for (size_t j = 0; j < _samples; j++)
		{
			oFile << j + 1 << "," << runtime.col(i)[j] << "," << rms.col(i)[j] << "," << fitness.col(i)[j] << std::endl;
		}
	}

	//close file
	oFile.close();
}
//main 
int main(int argc, char** argv)
{
	//healpix
	pcl::PointCloud<pcl::PointXYZ> HEALPixcloud;
	std::string cloudpath = "G:/PROJECT/test1/build/Debug/HEALPix.pcd";
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(cloudpath, HEALPixcloud) == -1)
	{
		PCL_ERROR("Couldn't read the %s th file \n", cloudpath);
	}
	
	int samples = 1;
	Eigen::Matrix<float, Eigen::Dynamic, 3> anglerandm = Eigen::MatrixXf::Random(samples, 3) * M_PI;
	Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f Rtransform = Eigen::Matrix4f::Identity();
	Eigen::Vector3f rand_sin, rand_cos;

	//result data structure
	Results Results1(samples, 2);
	Results1.angles = anglerandm;
	
	pcl::PointCloud<pcl::PointNormal> inputcloud;
	pcl::PointCloud<pcl::PointNormal> targetcloud;
	pcl::PointCloud<pcl::PointNormal> Rtargetcloud;
	pcl::PointCloud<pcl::PointNormal>::Ptr inputcloudPtr;
	pcl::PointCloud<pcl::PointNormal>::Ptr targetcloudPtr;

	pcl::PointCloud<pcl::PointXYZ> inputcloudnormals;
	pcl::PointCloud<pcl::PointXYZ> targetcloudnormals;
	
	RotationEstmtn *rotationestmtn = NULL;
	//load pointcloud
	cloudpath = "G:/PROJECT/test1/build/Debug/bladeNormalASCII.pcd";
	if (pcl::io::loadPCDFile<pcl::PointNormal>(cloudpath, inputcloud) == -1)
	{
		PCL_ERROR("Couldn't read the %s th file \n", cloudpath);
	}

	//get normals coordinates of input cloud
	inputcloudnormals.width = inputcloud.width;
	inputcloudnormals.height = inputcloud.height;
	targetcloudnormals.width = inputcloudnormals.width;
	targetcloudnormals.height = inputcloudnormals.height;
	inputcloudnormals.points.resize(inputcloud.points.size());
	for (size_t i = 0; i < inputcloudnormals.points.size(); ++i)
	{
		inputcloudnormals.points[i].x = inputcloud.points[i].normal_x;
		inputcloudnormals.points[i].y = inputcloud.points[i].normal_y;
		inputcloudnormals.points[i].z = inputcloud.points[i].normal_z;
	}
	//samples
	for (size_t j = 0; j < samples; j++)
	{
		//construct target point cloud
		rand_sin = anglerandm.row(j).array().sin();
		rand_cos = anglerandm.row(j).array().cos();
		transform.block(0, 0, 3, 3) =
			(Eigen::MatrixXf(3, 3) << 1, 0, 0, 0, rand_cos[0], rand_sin[0], 0, -rand_sin[0], rand_cos[0]).finished() *
			(Eigen::MatrixXf(3, 3) << rand_cos[1], 0, -rand_sin[1], 0, 1, 0, rand_sin[1], 0, rand_cos[1]).finished() *
			(Eigen::MatrixXf(3, 3) << rand_cos[2], rand_sin[2], 0, -rand_sin[2], rand_cos[2], 0, 0, 0, 1).finished();
		pcl::transformPointCloud(inputcloud, targetcloud, transform);
		pcl::transformPointCloud(inputcloudnormals, targetcloudnormals, transform);

		//rotation estimation
		Results1.watch.reset();
		Rtransform.setIdentity();
		rotationestmtn = new RotationEstmtn(inputcloudnormals, targetcloudnormals, HEALPixcloud);
		rotationestmtn->compute();
		Rtransform.block(0, 0, 3, 3) = rotationestmtn->M;
		//key variables estimation
		Results1.runtime(j, 0) = Results1.watch.getTime();
		Results1.iter(j, 0) = rotationestmtn->iter;
		Results1.fitness(j, 0) = rotationestmtn->fitness;
		pcl::transformPointCloud(targetcloud, Rtargetcloud, Rtransform);
		Results1.rms(j, 0) = Results1.RMS(inputcloud, Rtargetcloud);

		//delete object
		delete rotationestmtn;

	}

	//write results into *.CSV file
	Results1.outputCSV("MyAlgorithm");
	
	
	std::cout << "���������������";
	std::cin.clear();
	std::cin.sync();
	std::cin.get();
	return 0;
}
/*
//HEALPIX SAMPLING TEST
std::cout << "cloud->width" << sampling.cloud->width << "cloud->height" << sampling.cloud->height << "X" << sampling.cloud->points[0].x << "y" << sampling.cloud->points[0].y << "z" << sampling.cloud->points[0].z << sampling.kdtree.getInputCloud()->width<< "cols" << sampling.pts(3,0) << sampling.pts(3, 1) << std::endl;

//FTR
std::cout << ftrtest.density << std::endl;
//print normals into file
std::ofstream file("G:/PROJECT/test1/build/Debug/normals.txt");
if (file.is_open())
{
file << "Here is the matrix m:\n" << ftrtest.normals << '\n';
}

healpixSampling sampling(16);
sampling.generate();
//rotation matrix
Eigen::Matrix3f M;
M << 0.0000, 0.0000, -1.0000, 0.9135, 0.4067, 0.0000, 0.4067, -0.9135, -0.0000;
//get features
Ftr ftrtest(sampling, cloud);
ftrtest.compute();
//print results
pcl::PointXYZ point(0.5141, 0.8577, 0.0000);
//test sieve
Eigen::Matrix<float, Eigen::Dynamic, 2> normals;
normals.conservativeResize(cloud.size(), 2);
normals = cloud.getMatrixXfMap().adjoint().leftCols(2);
std::ofstream file2("G:/PROJECT/test1/build/Debug/normals2.txt");
if (file2.is_open())
{
file2 << "Here is the matrix m:\n" << cloud.getMatrixXfMap().adjoint() << '\n';
}
Sieve sieve(normals, normals.col(0).minCoeff(), normals.col(0).maxCoeff(), normals.col(1).minCoeff(), normals.col(1).maxCoeff(), (normals.col(0).maxCoeff() - normals.col(0).minCoeff()) / 20, 10);
sieve.compute();
std::cout << sieve.maxpoint << std::endl;
std::ofstream file1("G:/PROJECT/test1/build/Debug/normals1.txt");
if (file1.is_open())
{
file1 << "Here is the matrix m:\n" << normals << '\n';
}
*/