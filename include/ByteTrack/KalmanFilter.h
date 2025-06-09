#pragma once

#include "Eigen/Dense"

#include "ByteTrack/Rect.h"

namespace byte_track {
    class KalmanFilter {
    public:
        using DetectBox = Xyah<float>;

        using StateMean = Eigen::Matrix<float, 1, 8, Eigen::RowMajor>;
        using StateCov = Eigen::Matrix<float, 8, 8, Eigen::RowMajor>;

        using StateHMean = Eigen::Matrix<float, 1, 4, Eigen::RowMajor>;
        using StateHCov = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;

        /**
         * Kalman Filterの計算クラスのコンストラクタ
         * @param std_weight_position 位置ノイズの重み（0.05）
         * @param std_weight_velocity 速度ノイズの重み（0.00625）
         */
        explicit KalmanFilter(const float &std_weight_position = 1. / 20,
                              const float &std_weight_velocity = 1. / 160);

        /**
         * @fn
         * Kalman Filterの計算で利用する、平均（mean）と共分散（covariance）の初期値を作成します。
         * @param mean 平均初期値
         * @param covariance 共分散初期値
         * @param measurement 観測されたbounding box [bbox center x, bbox center y, aspect, height]
         */
        void initiate(StateMean &mean, StateCov &covariance, const DetectBox &measurement) const;

        /**
         * @fn
         * Kalman filterの予測ステップ（Predict Step）を実装した関数で、現在の状態を元に次の時刻の状態を予測する処理です。
         * @param mean 平均（現在を入力して、予測が出力されます。）
         * @param covariance 共分散（現在を入力して、予測が出力されます。）
         */
        void predict(StateMean &mean, StateCov &covariance);

        void update(StateMean &mean, StateCov &covariance, const DetectBox &measurement);

    private:
        float std_weight_position_;
        float std_weight_velocity_;

        /**
         * motion_mat_ は状態遷移行列 F（8×8行列）
         */
        Eigen::Matrix<float, 8, 8, Eigen::RowMajor> motion_mat_;

        /**
         * update_mat_ は観測行列 H（4×8行列）
         */
        Eigen::Matrix<float, 4, 8, Eigen::RowMajor> update_mat_;

        /**
         * @fn
         * 状態空間から観測空間への射影（projection）を行う処理です。
         * @param projected_mean 4次元の観測ベクトル [x, y, a, h]
         * @param projected_covariance 4x4の観測共分散
         * @param mean 平均（予測値）
         * @param covariance 共分散（予測値）
         */
        void project(StateHMean &projected_mean, StateHCov &projected_covariance,
                     const StateMean &mean, const StateCov &covariance);
    };
}
