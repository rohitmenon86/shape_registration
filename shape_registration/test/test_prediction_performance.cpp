// ROS
#include <ros/ros.h>
#include <ros/package.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>
#include<shape_registration_msgs/PredictShape.h>

// PCL
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/common.h>
#include <pcl/filters/passthrough.h>


#include <iostream>
#include<fstream>
#include <string>
#include <vector>
#include <filesystem>
namespace fs = std::filesystem;

using namespace std;


using std::string;

using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;
using namespace pcl::search;

using PointType = PointXYZRGB;
using Cloud = PointCloud<PointXYZRGB>;

template<typename TreeT, typename PointT>
float nearestDistance(const TreeT& tree, const PointT& pt)
{
  const int k = 1;
  std::vector<int> indices (k);
  std::vector<float> sqr_distances (k);

  tree.nearestKSearch(pt, k, indices, sqr_distances);

  return sqr_distances[0];
}

// compare cloudB to cloudA
// use threshold for identifying outliers and not considering those for the similarity
// a good value for threshold is 5 * <cloud_resolution>, e.g. 10cm for a cloud with 2cm resolution
template<typename CloudT>
float _similarity(CloudT& cloudA, CloudT& cloudB, float threshold)
{
  // compare B to A
  int num_outlier = 0;
  pcl::search::KdTree<typename CloudT::PointType> tree;
  tree.setInputCloud(cloudA.makeShared());
  auto sum = std::accumulate(cloudB.begin(), cloudB.end(), 0.0f, [&](auto current_sum, const auto& pt) {
    const auto dist = nearestDistance(tree, pt);

    if(dist < threshold)
    {
      return current_sum + dist;
    }
    else
    {
      num_outlier++;
      return current_sum;
    }
  });

  return sum / (cloudB.size() - num_outlier);
}

// comparing the clouds each way, A->B, B->A and taking the average
template<typename CloudT>
float similarity(CloudT& cloudA, CloudT& cloudB, float threshold = std::numeric_limits<float>::max())
{
    TicToc tt;
    tt.tic ();
    // compare B to A
    const auto similarityB2A = _similarity(cloudA, cloudB, threshold);
    // compare A to B
    const auto similarityA2B = _similarity(cloudB, cloudA, threshold);

    float dist = (similarityA2B * 0.5f) + (similarityB2A * 0.5f);

    print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" ms : ");
    print_info ("A->B: "); print_value ("%f", similarityB2A);
    print_info (", B->A: "); print_value ("%f", similarityA2B);
    print_info (", Chamfer Distance: "); print_value ("%f", dist);
    print_info (" ]\n");
    return dist;
}

class PredictionTester
{
    private:
    ros::NodeHandle m_nh;
    ros::ServiceClient m_client_shape_reg;
    ros::Publisher m_pub_full_cloud;
    ros::Publisher m_pub_partial_cloud;
    ros::Publisher m_pub_deformed_cloud;

    std::string m_root_dir;
    std::string m_input_dir;
    std::vector<int> m_pcd_files;

    public:

    PredictionTester()
    {
        m_client_shape_reg      = m_nh.serviceClient<shape_registration_msgs::PredictShape>("predict_shape");
        m_pub_full_cloud        = m_nh.advertise<sensor_msgs::PointCloud2>("full_cloud", 10);
        m_pub_partial_cloud     = m_nh.advertise<sensor_msgs::PointCloud2>("partial_cloud", 10);
        m_pub_deformed_cloud    = m_nh.advertise<sensor_msgs::PointCloud2>("deformed_cloud", 10);

        m_nh.param(std::string("root_dir"), m_root_dir, std::string("/home/rohit/data/pfuji-dataset/3-apple_point_clouds_pcd/"));
        m_nh.param(std::string("input_dir"), m_input_dir, std::string("downsampled/10mm/shifted/test/")); ///home/rohit/data/pfuji-dataset/3-apple_point_clouds_pcd/downsampled/10mm/shifted
        //m_nh.param("pcd_file_list", m_pcd_files, {21,22,23,24});

    }

    void publishPointCloud(ros::Publisher& pub, Cloud cloud)
    {
        sensor_msgs::PointCloud2 pc2;
        pcl::toROSMsg(cloud, pc2);
        pc2.header.stamp = ros::Time::now();
        pc2.header.frame_id = "base_link";
        pub.publish(pc2);

    }
    float computeDistanceHausdorff(Cloud &cloud_a, Cloud &cloud_b)
    {
        // Estimate
        TicToc tt;
        tt.tic ();

        print_highlight (stderr, "Computing ");

        // compare A to B
        pcl::search::KdTree<PointType> tree_b;
        tree_b.setInputCloud (cloud_b.makeShared ());
        float max_dist_a = -std::numeric_limits<float>::max ();
        for (const auto &point : cloud_a.points)
        {
            pcl::Indices indices (1);
            std::vector<float> sqr_distances (1);

            tree_b.nearestKSearch (point, 1, indices, sqr_distances);
            if (sqr_distances[0] > max_dist_a)
            max_dist_a = sqr_distances[0];
        }

        // compare B to A
        pcl::search::KdTree<PointType> tree_a;
        tree_a.setInputCloud (cloud_a.makeShared ());
        float max_dist_b = -std::numeric_limits<float>::max ();
        for (const auto &point : cloud_b.points)
        {
            pcl::Indices indices (1);
            std::vector<float> sqr_distances (1);

            tree_a.nearestKSearch (point, 1, indices, sqr_distances);
            if (sqr_distances[0] > max_dist_b)
            max_dist_b = sqr_distances[0];
        }

        max_dist_a = std::sqrt (max_dist_a);
        max_dist_b = std::sqrt (max_dist_b);

        float dist = std::max (max_dist_a, max_dist_b);

        print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" ms : ");
        print_info ("A->B: "); print_value ("%f", max_dist_a);
        print_info (", B->A: "); print_value ("%f", max_dist_b);
        print_info (", Hausdorff Distance: "); print_value ("%f", dist);
        print_info (" ]\n");

        return dist;
    }

    float computeDistanceChamfer(Cloud &cloud_a, Cloud &cloud_b)
    {
        return similarity(cloud_a, cloud_b);
    }

    void test()
    {
        for(int z = 1; z < 6; ++z)
        {
            std::ofstream myfile;
            std::string complete_input_path = m_root_dir + m_input_dir;
            myfile.open(complete_input_path + std::string("results_yfilter") + std::to_string(z*10) + std::string(".csv"));
            myfile<<"file_name, pred_time, hausdorff_dist, chamfer_dist\n";
            int file_number = 0;
            for (const auto & entry : fs::directory_iterator(complete_input_path))
            {
                std::string filename_with_ext = std::string(entry.path());
                ROS_WARN_STREAM("Opening pcd file: "<<filename_with_ext);
                if(filename_with_ext.find(".pcd")!=std::string::npos)
                {
                    file_number++;
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB> ());
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_partial (new pcl::PointCloud<pcl::PointXYZRGB> ());
                    pcl::PCDReader reader;
                    reader.read (filename_with_ext, *cloud); 
                    pcl::PointXYZRGB minPt, maxPt;
                    pcl::getMinMax3D (*cloud, minPt, maxPt);

                    pcl::PassThrough<pcl::PointXYZRGB> pass;
                    pass.setInputCloud (cloud);
                    pass.setFilterFieldName ("y");
                    pass.setFilterLimits (minPt.y, 0.5*(minPt.y + maxPt.y));
                    pass.filter (*cloud_partial);

                    shape_registration_msgs::PredictShape srv;
                    pcl::toROSMsg(*cloud_partial, srv.request.observed_point_cloud);
                
                    TicToc tt;
                    tt.tic();
                    double pred_time = 0;
                    bool success = false;
                    try
                    {
                        TicToc tt;
                        tt.tic();
                        success = m_client_shape_reg.call(srv);
                        pred_time = tt.toc ();
                        print_info ("Srv called, "); print_value ("%g", pred_time); print_info (" ms : ");
                    }
                    catch(const char* msg)
                    {
                        std::cerr << msg << '\n';
                    }
                    if(success)
                    {
                        ROS_WARN("Srv Success");
                        //print_info ("Srv called, "); print_value ("%g", tt.toc ()); print_info (" ms : ");
                        Cloud cloud_predicted;
                        std::cout<<srv.response.result_text<<std::endl;
                        ROS_WARN_STREAM("Res pc2 details: "<<srv.response.predicted_point_cloud.header);
                        srv.response.predicted_point_cloud.header.stamp = ros::Time::now();
                        srv.response.predicted_point_cloud.header.frame_id = "base_link";
                        for(int i = 0; i < 5; ++i)
                        {
                            publishPointCloud(m_pub_full_cloud, *cloud);
                            publishPointCloud(m_pub_partial_cloud, *cloud_partial);
                            m_pub_deformed_cloud.publish(srv.response.predicted_point_cloud);
                            sleep(0.1);
                        }
                        pcl::PCLPointCloud2 pcl_pc2;
                        pcl_conversions::toPCL(srv.response.predicted_point_cloud, pcl_pc2);
                        pcl::fromPCLPointCloud2(pcl_pc2, cloud_predicted);
                        //pcl::fromROSMsg(srv.response.predicted_point_cloud, cloud_predicted);
                        float dist1 = computeDistanceHausdorff(cloud_predicted, *cloud);
                        float dist2 = computeDistanceChamfer(cloud_predicted, *cloud);

                        myfile<<filename_with_ext<<","<<pred_time<<","<<dist1<<","<<dist2<<"\n";

                    }
                    sleep(1);
                }
            }
            myfile.close();

        }
    }
    
};

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "prediction_tester");

    PredictionTester pred_tester;
    pred_tester.test();
}