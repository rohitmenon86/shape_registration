// Categorical Shape Registration
// Author: Corbin Cogswell <corbincogswell@gmail.com>
//         Diego Rodriguez <rodriguez@ais.uni-bonn.de>

#pragma once

// QT
#ifndef Q_MOC_RUN
#include <QThread>
#include <QReadWriteLock>
#endif

#include <shape_registration/pca.hpp>
#include <shape_registration/solver.hpp>

// Eigen
#include <Eigen/Geometry>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;

enum SolverState
{
	SOLVER_NOT_INITIALISED = -1,
	SOLVER_RUNNING = 0,
	SOLVER_DONE_SUCCESS = 1,
	SOLVER_DONE_NO_CONVERGENCE = 2
};
class solver_thread : public QThread
{
	Q_OBJECT

public:
	solver_thread(QReadWriteLock* mutex);
	~solver_thread();

	void updateCanonicalMatrix(const MatrixXd &canonical_matrix);
	void updatePCA(const pca::ptr PCA);

	fit_iteration_callback* getCallback();

protected:
	void run();

private:
	void solve();
	void setSolverState(const SolverState& state)
	{
		m_mutex->lockForWrite();
		m_solver_state = state;
		m_mutex->unlock();
	}

	QReadWriteLock* m_mutex;

	pca::ptr m_PCA;

	MatrixXd m_canonical_matrix;
	MatrixXd m_observed_matrix;
	MatrixXd m_XStar;

	Eigen::Affine3d m_local_rigid;

	fit_iteration_callback* m_callback;

	// Config parameters
	const float m_sigma = 0.05;
	const int m_max_iterations = 40;
	const int m_cost_function = 2;
	const bool m_opt_sigma = false;

	SolverState m_solver_state;
	ceres::Solver::Summary m_solver_summary;

signals:
	void fitted(const MatrixXd& XStar, const Eigen::Affine3d& trans);

public slots:
	void fit(MatrixXd observed_matrix, const MatrixXd& XStar, const Eigen::Affine3d& trans);
	void halt();

public:
	SolverState getSolverState()
	{
		SolverState state;
		m_mutex->lockForRead();
		state = m_solver_state;
		m_mutex->unlock();
		return state;
	}	
	Eigen::Affine3d getLocalRigidTransform()
	{
		return m_local_rigid;
	}
	Eigen::MatrixXd getLatent()
	{
		return m_XStar;
	}
	ceres::Solver::Summary getSolverSummary()
	{
		return m_solver_summary;
	}
	
};
