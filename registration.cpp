#include <rgstrtn.h>

//healpixSampling methods
healpixSampling::healpixSampling(int n = 16): _n(n) {};

void
healpixSampling::generate()
{
	sampling();// 
	//matrix to point cloud
	cloud->width = 12 * _n ^ 2;
	cloud->height = 1;
	cloud->points.resize (cloud->width * cloud->height);
	for (size_t i = 0; i < cloud->points.size (); ++i)
	{
		cloud->points[i].x = pts(i, 0);
		cloud->points[i].y = pts(i, 1);
		cloud->points[i].z = pts(i, 2);
	}
	//build ketree ready to search
	kdtree.setInputCloud (cloud);
}

void
healpixSampling::sampling()
{
	std::ifstream csvfl("HEALPix.csv");
	std::vector<std::string> fields;
	std::string line, field;
	int i = 0;
	while (std::getline(csvfl, line))
	{
		std::istringstream sin(line);
		while (std::getline(sin, field, ','))
		{
			fields.push_back(field);
		}
		pts.row(i) << std::stod(fields[0]), std::stod(fields[1]), std::stod(fields[2]);
		i++;
	}
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
		normals = (M * normals.transpose()).transpose();
		//matrix to point cloud
		_InputCloud.width = normals.rows();
		_InputCloud.height = 1;
		_InputCloud.points.resize (_InputCloud.width * _InputCloud.height);
		for (size_t i = 0; i < _InputCloud.points.size (); ++i)
		{
			_InputCloud.points[i].x = normals(i, 0);
			_InputCloud.points[i].y = normals(i, 1);
			_InputCloud.points[i].z = normals(i, 2);
		}
		std::vector<int> pointIdxNKNSearch(_Kneighbours);
		std::vector<float> k_sqr_distances(_Kneighbours);

		intensity.resize(_healpiX.pts.rows());
		
		for(int i = 0; i < _InputCloud.points.size(); ++i)
		{
			_healpiX.kdtree.nearestKSearch(_InputCloud, i, _Kneighbours, pointIdxNKNSearch, k_sqr_distances);
			for (size_t j = 0; j < _Kneighbours; j++)
			{
				intensity[pointIdxNKNSearch[j]] ++;
				normals_index[pointIdxNKNSearch[j]].resize(normals_index[pointIdxNKNSearch[j]].size() + 1);
				normals_index[pointIdxNKNSearch[j]][normals_index[pointIdxNKNSearch[j]].size() - 1] = i;
			}
		}
	}

void
Ftr::backup()
{
	intensity0 = intensity;
	density0 = density;
	normals_index0 = normals_index;
	normals0 = normals;
}

void
Ftr::restore()
{
	intensity = intensity0;
	density = density0;
	normals_index = normals_index0;
	normals = normals0;
}

Eigen::Matrix<float, Eigen::Dynamic, 3>
Ftr::find(pcl::PointXYZ point, int Kpixels)
{
	Eigen::Matrix<float, Eigen::Dynamic, 3> nearestnormals;
	std::vector<int> pointIdxNKNSearch;
	std::vector<float> k_sqr_distances;
	_healpiX.kdtree.nearestKSearch(point, Kpixels, pointIdxNKNSearch, k_sqr_distances);
	for (size_t i = 0; i < pointIdxNKNSearch.size(); i++)
	{
		nearestnormals.row(i) = _healpiX.pts.row(pointIdxNKNSearch[i]);
	}
}

void
Ftr::compute()
{
	std::vector<int> pointIdxNKNSearch(_Kneighbours);
	std::vector<float> k_sqr_distances(_Kneighbours);

	intensity.resize(_healpiX.pts.rows());

	for (int i = 0; i < _InputCloud.points.size(); ++i)
	{
		_healpiX.kdtree.nearestKSearch(_InputCloud, i, _Kneighbours, pointIdxNKNSearch, k_sqr_distances);
		for (size_t j = 0; j < _Kneighbours; j++)
		{
			intensity[pointIdxNKNSearch[j]] ++;
			normals_index[pointIdxNKNSearch[j]].resize(normals_index[pointIdxNKNSearch[j]].size() + 1);
			normals_index[pointIdxNKNSearch[j]][normals_index[pointIdxNKNSearch[j]].size() - 1] = i;
		}
	}
	
	normals = _InputCloud.getMatrixXfMap(3, 4, 0);
	Eigen::Matrix<int, Eigen::Dynamic, 1> density_tmp (intensity.data());
	density = density_tmp;
}


//RotationEstmtn methods
RotationEstmtn::RotationEstmtn(const pcl::PointCloud<pcl::PointXYZ> &InputCloud1, 
	const pcl::PointCloud<pcl::PointXYZ> &InputCloud2, int iterMax, int stall, int Kneighbours, 
	int steps, int blocks, float tolerance)
	: _InputCloud1(InputCloud1), _InputCloud2(InputCloud2), _iterMax(iterMax), _stall(stall),
	_Kneighbours(Kneighbours), _steps(steps), _blocks(blocks), _tolerance(tolerance)
{
}

RotationEstmtn::~RotationEstmtn()
{
	delete ftr_model;
	delete ftr_data;
	delete ftr_datatmp;
}

void
RotationEstmtn::compute()
{
	//
	Eigen::Vector3f rand, rand_sin, rand_cos;
	Eigen::Matrix3f T_rot, T_tmp;
	rand = Eigen::MatrixXf::Random(1,3) * M_PI;
	rand_sin =  rand.array().sin();
	rand_cos = rand.array().cos();
	T_rot = (Eigen::MatrixXf(3, 3) << 1, 0, 0, 0, rand_cos[1], rand_sin[1], 0, -rand_sin[1], rand_cos[1]).finished() *
					(Eigen::MatrixXf(3, 3) << rand_cos[2], 0, -rand_sin[1], 0, 1, 0, rand_sin[2], 0, rand_cos[2]).finished() *
					(Eigen::MatrixXf(3, 3) << rand_cos[3], rand_sin[3], 0, -rand_sin[3], rand_cos[3], 0, 0, 0, 1).finished();

	M.setIdentity();


	healpixSampling healpiX (16);
	healpiX.generate();

	ftr_model = new Ftr(healpiX, _InputCloud1);
	ftr_model->compute();
	ftr_data = new Ftr(healpiX, _InputCloud2);
	ftr_data->compute();
	//rotation
	iter = 0;
	int stall = 0;
	ftr_datatmp = ftr_data;//copying a object, extra attention
	float fitness_tmp;
	fitness = 0;
	//iteration
	while(fitness < 1-_tolerance && iter < _iterMax)
	{
		T_tmp = Rotationest();
		ftr_datatmp->refresh(T_tmp);
		fitness_tmp = ftr_model->density.dot(ftr_datatmp->density) / (ftr_model->density.norm() * ftr_datatmp->density.norm());
		
		stall ++;
		if(fitness <= fitness_tmp)
		{
			if(fitness_tmp > fitness * 1.005) stall --;
			fitness = fitness_tmp;
			M = T_tmp * M;
		}
		else
		{
			ftr_datatmp->restore();
		}

		if (stall >= 3 && fitness_tmp < 1-_tolerance)
		{
			M = T_rot * M;
			ftr_datatmp->refresh(T_rot);
			stall = 0;
		}

		iter ++;
	}
}

Eigen::Matrix3f
RotationEstmtn::Rotationest()
{
	Eigen::Matrix <int, 12, 2> ind_md, ind_dd;
	Eigen::Matrix <float, Eigen::Dynamic, 5> pts_md, pts_dd;
	int pixel_size = ftr_model->density.size() / 12;

	for (int i = 0; i < 12; ++i)
	{
		ind_md(i,1) = ftr_model->density.segment(pixel_size*i,pixel_size).maxCoeff(&ind_md(i,2));
		ind_md(i,2) = ind_md(i,2) + pixel_size * i;
		ind_dd(i,1) = ftr_datatmp->density.segment(pixel_size*i,pixel_size).maxCoeff(&ind_dd(i,2));
		ind_dd(i,2) = ind_dd(i,2) + pixel_size * i;

		pts_md.row(i) << ftr_model->normals.row(ind_md(i,2)), ind_md(i,1), ind_md(i,2);
		pts_dd.row(i) << ftr_datatmp->normals.row(ind_dd(i,2)), ind_dd(i,1), ind_dd(i,2);
	}
	//sort

	//
	int i = 1;
	while (abs(pts_md.row(0).head(3).dot(pts_md.row(i).head(3))) > 0.94)
		i ++;
	pts_md.row(1) = pts_md.row(i);
	
	//
	for (i = 1; i < 12; ++i)
	{
		if (abs((pts_dd.row(i).head(3) - pts_dd.row(0).head(3)).norm()
		 - (pts_md.row(1).head(3) - pts_md.row(0).head(3)).norm()) < 0.1)
		{
			pts_dd.row(2) = pts_dd.row(i);
			break;
		}
	}

	//
	Eigen::Matrix<float, Eigen::Dynamic, 2> normalsXY_md, normalsXY_dd;
	Sieve *sieve_md = NULL, *sieve_dd = NULL;
	float z_md, z_dd;
	Eigen::Vector3f x_md, y_md, x_dd, y_dd;
	for (int i = 0; i < 2; ++i)
	{	
		z_md = -0.5 / pts_md(i, 3)*(pts_md(i, 1) + pts_md(i, 2));
		z_dd = -0.5 / pts_dd(i, 3)*(pts_dd(i, 1) + pts_dd(i, 2));
		x_md = ((Eigen::MatrixXf(1, 3) << 0.5, 0.5, z_md).finished() / (Eigen::MatrixXf(1, 3) << 0.5, 0.5, z_md).finished().norm());
		y_md = pts_md.row(i).head(3).cross(x_md);
		x_dd = ((Eigen::MatrixXf(1, 3) << 0.5, 0.5, z_dd).finished() / (Eigen::MatrixXf(1, 3) << 0.5, 0.5, z_dd).finished().norm());
		y_dd = pts_dd.row(i).head(3).cross(x_dd);
		
		//project to a plane perpendicular to pts_md.row(i).head(3)
		normalsXY_md = ((Eigen::MatrixXf(2, 3) << x_md, y_md).finished() * ftr_model->find(pcl::PointXYZ(pts_md(i, 0), pts_md(i, 1), pts_md(i, 2)), _Kneighbours).adjoint()).adjoint();
		normalsXY_dd = ((Eigen::MatrixXf(2, 3) << x_dd, y_dd).finished() * ftr_datatmp->find(pcl::PointXYZ(pts_dd(i, 0), pts_dd(i, 1), pts_dd(i, 2)), _Kneighbours).adjoint()).adjoint();
		
		//
		sieve_md = new Sieve (normalsXY_md, normalsXY_md.col(0).minCoeff(), normalsXY_md.col(0).maxCoeff(), normalsXY_md.col(1).minCoeff(),
		 					normalsXY_md.col(1).maxCoeff(), normalsXY_md.col(0).maxCoeff() - normalsXY_md.col(0).minCoeff() / _steps, _blocks);
		sieve_dd = new Sieve (normalsXY_dd, normalsXY_dd.col(0).minCoeff(), normalsXY_dd.col(0).maxCoeff(), normalsXY_dd.col(1).minCoeff(),
							normalsXY_dd.col(1).maxCoeff(), normalsXY_dd.col(0).maxCoeff() - normalsXY_dd.col(0).minCoeff() / _steps, _blocks);
		sieve_md->compute();
		sieve_dd->compute();
		
		pts_md.row(i).head(3) = sieve_md->maxpoint[0] * x_md + sieve_md->maxpoint[1] * y_md + sqrt(1-pow(sieve_md->maxpoint[0], 2)- pow(sieve_md->maxpoint[1], 2)) * pts_md.row(i).head(3);
		pts_dd.row(i).head(3) = sieve_dd->maxpoint[0] * x_dd + sieve_dd->maxpoint[1] * y_dd + sqrt(1-pow(sieve_dd->maxpoint[0], 2) - pow(sieve_dd->maxpoint[1], 2)) * pts_dd.row(i).head(3);
		
		delete sieve_dd;
		delete sieve_md;
	}

	//rotation vectors construction
	Eigen::Vector3f r1 = (pts_dd.row(0).head(3).cross(pts_md.row(0).head(3))/pts_dd.row(0).head(3).cross(pts_md.row(0).head(3)).norm());
	float theta1 = acos(pts_dd.row(0).head(3).dot(pts_dd.row(0).head(3)));
	Eigen::AngleAxisf R1(theta1, r1);
	
	Eigen::Vector3f r2 = pts_md.row(0).head(3);
	Eigen::Vector3f a = pts_md.row(1).head(3) - pts_md.row(1).head(3).dot(pts_md.row(0).head(3)) * r2;
	//Eigen::MatrixXf c = (R1.matrix() * pts_dd.row(1).head(3).adjoint()).adjoint() - (R1.matrix() * pts_dd.row(1).head(3).adjoint()).adjoint().dot(r2) * r2;( c.data()[0], c.data()[1], c.data()[2])
	//Eigen::MatrixXf c (1, 2, 3);
	Eigen::Vector3f b = (R1.matrix() * pts_dd.row(1).head(3).adjoint()) - ((R1.matrix() * pts_dd.row(1).head(3).adjoint()).dot(r2) * r2).adjoint();
	float theta2 = acos(a.dot(b)) / (a.norm() * b.norm());
	Eigen::AngleAxisf R2(theta2, r2);

	M = R2.matrix() * R1.matrix();
	
}

//class Sieve methods
Sieve::Sieve(Eigen::Matrix<float, Eigen::Dynamic, 2> normals, float xmin, float xmax, float ymin, float ymax, float radius, int grids):
			_normals(normals), _xmin(xmin), _xmax(xmax), _ymin(ymin), _ymax(ymax), _radius(radius), _grids(grids) {};

void
Sieve::compute()
{
	for (int i = 0; i < _grids; ++i)
		{
		pixels.block(i, 0, _grids, 1) = Eigen::MatrixXf(_grids, 1, ((_xmax - _xmin) / (_grids - 1) * i + _xmin));
			pixels.block(i, 1, _grids, 1) = Eigen::VectorXf::LinSpaced(_grids, _ymin, _ymax);
		}
	pixels.col(2) = Eigen::Matrix<float, Eigen::Dynamic, 1>::Zero(_grids^2, 1);
	pixels.col(2) = Eigen::Matrix<float, Eigen::Dynamic, 1>::Zero(_grids^2, 1);

	//
	cloud.width = _grids ^ 2;
	cloud.height = 1;
	cloud.points.resize (cloud.width * cloud.height);
	for (size_t i = 0; i < cloud.points.size (); ++i)
	{
		cloud.points[i].x = pixels(i, 0);
		cloud.points[i].y = pixels(i, 1);
		cloud.points[i].z = pixels(i, 2);
	}
	normalsCloud->width = _normals.rows();
	normalsCloud->height = 1;
	normalsCloud->points.resize (normalsCloud->width * normalsCloud->height);
	for (int i = 0; i < _normals.rows(); ++i)
	{
		normalsCloud->points[i].x = _normals(i, 0);
		normalsCloud->points[i].y = _normals(i, 1);
		normalsCloud->points[i].z = _normals(i, 2);
		
	}

	kdtree.setInputCloud (normalsCloud);
	std::vector<int> pointIdxRadiusSearch;
	std::vector<float> k_sqr_distances;
	for(int i = 0; i < cloud.points.size(); ++i)
	{
		kdtree.radiusSearch(cloud, i, _radius, pointIdxRadiusSearch, k_sqr_distances);
		intensity[i] = (int) pointIdxRadiusSearch.size();
	}

	int max_index;
	intensity.maxCoeff(&max_index);
	maxpoint = pixels.row(max_index).head(2);
}

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
	for (size_t i = 0; i < inputcloud.points.size(); ++i)
	{
		RMS = RMS + pow(inputcloud.points[i].x - targetcloud.points[i].x, 2) + pow(inputcloud.points[i].y - targetcloud.points[i].y, 2) + pow(inputcloud.points[i].y - targetcloud.points[i].y, 2);
	}
	RMS = sqrt(RMS / inputcloud.points.size());
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
		oFile << i+1 << "," << angles.row(i)[0] << "," << angles.row(i)[1] << "," << angles.row(i)[2] << std::endl;
	}
	for (size_t i = 0; i < _argc; i++)
	{
		oFile << "The" << i+1 << "-th point cloud" << std::endl;
		oFile << "No." << "," << "Run Time(ms)" << "," << "RMS" << "," << "Corr" << "," << "Iterations" << std::endl;
		for (size_t j = 0; j < _samples; j++)
		{
			oFile << j + 1 << "," << runtime.col(i)[j] << "," << rms.col(i)[j] << "," << fitness.col(i)[j] << std::endl;
		}
	}

	//close file
	oFile.close();
}
