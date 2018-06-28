#include <rgstrtn.h>
int main(int argc, char** argv)
{
	if (argc == 0 )
	{
		std::cerr << "Input at least ONE *.pcd file";
		exit(0);
	}
	int samples = 50;
	Eigen::Matrix<float, Eigen::Dynamic, 3> anglerandm = Eigen::MatrixXf::Random(samples, 3) * M_PI;
	Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f Rtransform = Eigen::Matrix4f::Identity();
	Eigen::Vector3f rand_sin, rand_cos;

	//result data structure
	Results Results1(samples, argc);
	Results Results2(samples, argc);
	Results Results3(samples, argc);
	Results1.angles = anglerandm;
	Results2.angles = anglerandm;
	Results3.angles = anglerandm;

	pcl::PointCloud<pcl::PointNormal> inputcloud;
	pcl::PointCloud<pcl::PointNormal> targetcloud;
	pcl::PointCloud<pcl::PointNormal> Rtargetcloud;
	pcl::PointCloud<pcl::PointNormal>::Ptr inputcloudPtr;
	pcl::PointCloud<pcl::PointNormal>::Ptr targetcloudPtr;

	pcl::PointCloud<pcl::PointXYZ> inputcloudnormals;
	pcl::PointCloud<pcl::PointXYZ> targetcloudnormals;

	//Sac-ia
	Sac_ia *sac_ia = NULL;
	pcl::PointCloud<pcl::PointXYZ>::Ptr inputcloudXYZPtr;
	pcl::PointCloud<pcl::PointXYZ>::Ptr targetcloudXYZPtr;

	pcl::PointCloud<pcl::PointXYZ> inputcloudXYZ;
	pcl::PointCloud<pcl::PointXYZ> targetcloudXYZ;

	//PCA
	Pcaalign* Pca;


	RotationEstmtn *rotationestmtn = NULL;

	for (size_t i = 1; i < argc; i++)
	{
		if (pcl::io::loadPCDFile<pcl::PointNormal>(argv[i], inputcloud) == -1)
		{
			PCL_ERROR("Couldn't read the %d th file \n", i);
			return (-1);
		}
		anglerandm = Eigen::MatrixXf::Random(samples, 3) * M_PI;

		//get normals coordinates of input cloud
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
			rand_sin = anglerandm.row(i).array().sin();
			rand_cos = anglerandm.row(i).array().cos();
			transform.block(0,0,3,3) = 
				(Eigen::MatrixXf(3, 3) << 1, 0, 0, 0, rand_cos[1], rand_sin[1], 0, -rand_sin[1], rand_cos[1]).finished() *
				(Eigen::MatrixXf(3, 3) << rand_cos[2], 0, -rand_sin[1], 0, 1, 0, rand_sin[2], 0, rand_cos[2]).finished() *
				(Eigen::MatrixXf(3, 3) << rand_cos[3], rand_sin[3], 0, -rand_sin[3], rand_cos[3], 0, 0, 0, 1).finished();
			pcl::transformPointCloud(inputcloud, targetcloud, transform);
			//get normals coordinates of target cloud
			targetcloudnormals.points.resize(inputcloud.points.size());
			for (size_t i = 0; i < targetcloudnormals.points.size(); ++i)
			{
				targetcloudnormals.points[i].x = inputcloud.points[i].normal_x;
				targetcloudnormals.points[i].y = inputcloud.points[i].normal_y;
				targetcloudnormals.points[i].z = inputcloud.points[i].normal_z;
			}

			//	//////////////////////////////////////////////////////////// Algorithm 1
			//rotation estimation
			Results1.watch.reset();
			Rtransform.setIdentity();
			rotationestmtn = new RotationEstmtn(inputcloudnormals, targetcloudnormals);
			rotationestmtn->compute();
			Rtransform.block(0, 0, 3, 3) = rotationestmtn->M;
			//key variables estimation
			Results1.runtime(j, i) = Results1.watch.getTime();
			Results1.iter(j, i) = rotationestmtn->iter;
			Results1.fitness(j, i) = rotationestmtn->fitness;
			pcl::transformPointCloud(targetcloud, Rtargetcloud, Rtransform);
			Results1.rms(j, i) = Results1.RMS(inputcloud, Rtargetcloud);

			//delete object
			delete rotationestmtn;
			//	//////////////////////////////////////////////////////////// Algorithm 1

			//	//////////////////////////////////////////////////////////// Algorithm 2
			//PCA alignment
			Results2.watch.reset();
			Rtransform.setIdentity();
			*inputcloudXYZPtr = inputcloudXYZ;
			*targetcloudXYZPtr = targetcloudXYZ;
			Pca = new Pcaalign(inputcloudXYZPtr, targetcloudXYZPtr);
			Pca->compute();
			Rtransform.block(0, 0, 3, 3) = Pca->M;

			//key variables estimation
			Results2.runtime(j, i) = Results2.watch.getTime();
			Results2.iter(j, i) = 0;
			Results2.fitness(j, i) = 0;
			pcl::transformPointCloud(targetcloud, Rtargetcloud, Rtransform);
			Results2.rms(j, i) = Results2.RMS(inputcloud, Rtargetcloud);

			//	//////////////////////////////////////////////////////////// Algorithm 2

			//	//////////////////////////////////////////////////////////// Algorithm 3
			//SAC-IAÅä×¼
			pcl::copyPointCloud(inputcloudXYZ, inputcloud);
			pcl::copyPointCloud(targetcloudXYZ, targetcloud);
			*inputcloudXYZPtr = inputcloudXYZ;
			*targetcloudXYZPtr = targetcloudXYZ;
			Results3.watch.reset();
			sac_ia = new Sac_ia(inputcloudXYZPtr, targetcloudXYZPtr);

			//get results
			Results3.runtime(j, i) = Results3.watch.getTime();
			Results2.iter(j, i) = 0;
			Results2.fitness(j, i) = 0;
			Results3.rms(j, i) = sac_ia->Results.fitness_score;

			//delete 
			delete sac_ia;
			//	//////////////////////////////////////////////////////////// Algorithm 3

		}

		//write results into *.CSV file
		Results1.outputCSV("MyAlgorithm");
		Results2.outputCSV("ICP");
		Results3.outputCSV("EGI");
	}
}
/*
//rotation estimation
*inputcloudPtr = inputcloud;
*targetcloudPtr = targetcloud;
Results2.watch.reset();
icp.setInputSource(inputcloudPtr);
icp.setInputTarget(targetcloudPtr);
icp.setTransformationEpsilon(1e-10);
icp.setEuclideanFitnessEpsilon(1e-3);
icp.setMaximumIterations(50);
icp.align(Rtargetcloud);

//key variables estimation
Results2.runtime(j, i) = Results2.watch.getTime();
Results2.iter(j, i) = 50;
Results2.fitness(j, i) = 0;
Results2.rms(j, i) = icp.getFitnessScore();

//delete object
delete rotationestmtn;
*/