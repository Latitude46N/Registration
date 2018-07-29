#include <rgstrtn.h>

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

//Ftr methods
Ftr::Ftr(const healpixSampling &healpiX, const pcl::PointCloud<pcl::PointXYZ> &InputCloud, int Kneighbours)
	:_healpiX(healpiX), _InputCloud(InputCloud), _Kneighbours(Kneighbours)
{
}

void
Ftr::refresh(const Eigen::Matrix3f &M)
{
	backup();
	
	Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
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
Ftr::find(int index, int Kpixels)
{
	Eigen::Matrix<float, Eigen::Dynamic, 3> nearestnormals;
	std::vector<int> pointIdxNKNSearch;
	std::vector<int> normalsmapIdx;
	std::vector<int> normalsIdx;
	std::vector<float> k_sqr_distances;
	_healpiX.kdtree.nearestKSearch(_healpiX.kdtree.getInputCloud()->points[index], Kpixels, pointIdxNKNSearch, k_sqr_distances);

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

// Sieve methods
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
	while (fitness < 1 - _tolerance && iter < _iterMax)
	{
		T_tmp = Rotationest();
		ftr_data->refresh(T_tmp);
		fitness_tmp = (float)ftr_model->density.dot(ftr_data->density) / (float)(ftr_model->density.norm() * ftr_data->density.norm());

		stall++;
		if (fitness * 1.005 < fitness_tmp)
		{
			stall = 0;
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
			//fitness = (float)ftr_model->density.dot(ftr_data->density) / (float)(ftr_model->density.norm() * ftr_data->density.norm());
		}
		
		iter++;
	}
}

Eigen::Matrix3f
RotationEstmtn::Rotationest()
{
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

		//project to a plane perpendicular to pts_md.row(i).head(3)
		normalsXY_md.resize(0, 0);
		normalsXY_dd.resize(0, 0);
		normalsXY_md = ((Eigen::MatrixXf(2, 3) << x_md.adjoint(), y_md.adjoint()).finished() * ftr_model->find(pts_md(i, 4), _Kneighbours).adjoint()).adjoint();
		normalsXY_dd = ((Eigen::MatrixXf(2, 3) << x_dd.adjoint(), y_dd.adjoint()).finished() * ftr_data->find(pts_dd(i, 4), _Kneighbours).adjoint()).adjoint();
		
		//
		sieve_md = new Sieve(normalsXY_md, normalsXY_md.col(0).minCoeff(), normalsXY_md.col(0).maxCoeff(), normalsXY_md.col(1).minCoeff(),
			normalsXY_md.col(1).maxCoeff(), (normalsXY_md.col(0).maxCoeff() - normalsXY_md.col(0).minCoeff()) / _steps, _blocks);
		sieve_dd = new Sieve(normalsXY_dd, normalsXY_dd.col(0).minCoeff(), normalsXY_dd.col(0).maxCoeff(), normalsXY_dd.col(1).minCoeff(),
			normalsXY_dd.col(1).maxCoeff(), (normalsXY_dd.col(0).maxCoeff() - normalsXY_dd.col(0).minCoeff()) / _steps, _blocks);
		sieve_md->compute();
		sieve_dd->compute();

		pts_md.row(i).head(3) = sieve_md->maxpoint[0] * x_md.adjoint() + sieve_md->maxpoint[1] * y_md.adjoint() + sqrt(1 - pow(sieve_md->maxpoint[0], 2) - pow(sieve_md->maxpoint[1], 2)) * pts_md.row(i).head(3);
		pts_dd.row(i).head(3) = sieve_dd->maxpoint[0] * x_dd.adjoint() + sieve_dd->maxpoint[1] * y_dd.adjoint() + sqrt(1 - pow(sieve_dd->maxpoint[0], 2) - pow(sieve_dd->maxpoint[1], 2)) * pts_dd.row(i).head(3);

		delete sieve_dd;
		delete sieve_md;
	}
	
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
	float theta2 = acos(a.dot(b) / (a.norm() * b.norm()));
	Eigen::AngleAxisf R2(theta2, r2);
	T = R2.matrix() * R1.matrix();

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

//Results methods
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
		RMS = RMS + pow((inputcloudM.row(i)-targetcloudM.row(i)).norm(), 2);
	}
	RMS = sqrt(RMS / (float)inputcloud.points.size());
	return RMS;
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
			oFile << j + 1 << "," << runtime.col(i)[j] << "," << rms.col(i)[j] << "," << fitness.col(i)[j] << "," << iter.col(i)[j] << std::endl;
		}
	}

	//close file
	oFile.close();
}