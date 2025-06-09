#include "ByteTrack/KalmanFilter.h"

/**
 * @brief Kalman Filterの計算クラスのコンストラクタ
 * 等速直線運動モデルを採用している。
 *
 * motion_mat_ は状態遷移行列 F（8×8行列）
 *   F = [1, 0, 0, 0, 1, 0, 0, 0]  // x_new  = x_old  + vx*dt
 *       [0, 1, 0, 0, 0, 1, 0, 0]  // y_new  = y_old  + vy*dt
 *       [0, 0, 1, 0, 0, 0, 1, 0]  // a_new  = a_old  + va*dt
 *       [0, 0, 0, 1, 0, 0, 0, 1]  // h_new  = h_old  + vh*dt
 *       [0, 0, 0, 0, 1, 0, 0, 0]  // vx_new = vx_old (速度は一定)
 *       [0, 0, 0, 0, 0, 1, 0, 0]  // vy_new = vy_old
 *       [0, 0, 0, 0, 0, 0, 1, 0]  // va_new = va_old
 *       [0, 0, 0, 0, 0, 0, 0, 1]  // vh_new = vh_old
 * update_mat_ は観測行列 H（4×8行列）
 *   H = [1, 0, 0, 0, 0, 0, 0, 0]  // x成分を抽出
 *       [0, 1, 0, 0, 0, 0, 0, 0]  // y成分を抽出
 *       [0, 0, 1, 0, 0, 0, 0, 0]  // a成分を抽出
 *       [0, 0, 0, 1, 0, 0, 0, 0]  // h成分を抽出
 * @param std_weight_position 位置ノイズの重み（default: 0.05）
 * @param std_weight_velocity  速度ノイズの重み（default: 0.00625）
 */
byte_track::KalmanFilter::KalmanFilter(const float &std_weight_position,
                                       const float &std_weight_velocity) : std_weight_position_(std_weight_position),
                                                                           std_weight_velocity_(std_weight_velocity) {
    constexpr Eigen::Index ndim = 4;
    constexpr float dt = 1; // 時間ステップ = 1フレーム

    motion_mat_ = Eigen::MatrixXf::Identity(8, 8);
    update_mat_ = Eigen::MatrixXf::Identity(4, 8);

    for (Eigen::Index i = 0; i < ndim; i++) {
        motion_mat_(i, ndim + i) = dt;
    }
}

/**
 * @fn
 * Kalman Filterの計算で利用する、平均（mean）と共分散（covariance）の初期値を作成します。
 * @param mean 平均 最初に観測された速度0の8次元ベクトルとして初期化します。 [x, y, a, h, 0, 0, 0, 0]
 * @param covariance 共分散 観測データの高さ（大きさ）に比例したノイズを設定し、共分散行列を初期化します。
 * @param measurement 観測されたbounding box [bbox center x, bbox center y, aspect, height]
 */
void byte_track::KalmanFilter::initiate(StateMean &mean, StateCov &covariance, const DetectBox &measurement) const {
    mean.block<1, 4>(0, 0) = measurement.block<1, 4>(0, 0);
    mean.block<1, 4>(0, 4) = Eigen::Vector4f::Zero();

    // measurement[3] つまり bbox の height（大きさ）をもとにノイズ（標準偏差）を設定します。
    StateMean std;
    std(0) = 2 * std_weight_position_ * measurement[3]; // 中心x座標のノイズ
    std(1) = 2 * std_weight_position_ * measurement[3]; // 中心y座標のノイズ
    std(2) = 1e-2; // アスペクト比のノイズ
    std(3) = 2 * std_weight_position_ * measurement[3]; // 高さのノイズ
    std(4) = 10 * std_weight_velocity_ * measurement[3]; // vxのノイズ
    std(5) = 10 * std_weight_velocity_ * measurement[3]; // vyのノイズ
    std(6) = 1e-5; // アスペクト比変化速度のノイズ
    std(7) = 10 * std_weight_velocity_ * measurement[3]; // 高さ変化速度のノイズ

    const StateMean tmp = std.array().square(); // 標準偏差を分散に変換（二乗）
    covariance = tmp.asDiagonal(); // 対角行列として共分散行列を作成
}

/**
 * @fn
 * Kalman filterの予測ステップ（Predict Step）を実装した関数で、現在の状態を元に次の時刻の状態を予測する処理です。
 * 現在の平均の height（大きさ）をもとにノイズ（標準偏差）が計算されて、それを元に予測されます。
 * @param mean 平均（現在を入力して、予測が出力されます。）
 * @param covariance 共分散（現在を入力して、予測が出力されます。）
 */
void byte_track::KalmanFilter::predict(StateMean &mean, StateCov &covariance) {
    StateMean std;
    std(0) = std_weight_position_ * mean(3); // 中心x座標のノイズ
    std(1) = std_weight_position_ * mean(3); // 中心y座標のノイズ
    std(2) = 1e-2; // アスペクト比のノイズ
    std(3) = std_weight_position_ * mean(3); // 高さのノイズ
    std(4) = std_weight_velocity_ * mean(3); // vxのノイズ
    std(5) = std_weight_velocity_ * mean(3); // vyのノイズ
    std(6) = 1e-5; // アスペクト比変化速度のノイズ
    std(7) = std_weight_velocity_ * mean(3); // 高さ変化速度のノイズ

    const StateMean tmp = std.array().square(); // 標準偏差を分散に変換（二乗）
    const StateCov motion_cov = tmp.asDiagonal(); // 対角行列として共分散行列を作成

    // 状態予測: x_k|k-1 = F * x_k-1|k-1
    mean = motion_mat_ * mean.transpose();

    // 共分散予測: P_k|k-1 = F * P_k-1|k-1 * F^T + Q（ノイズ）
    covariance = motion_mat_ * covariance * (motion_mat_.transpose()) + motion_cov;
}

/**
 * @fn
 * Kalman filterの更新ステップ（Update Step）を実装した関数で、予測した状態を実際の観測結果で修正する処理です。
 * この更新ステップにより、トラックの状態がより正確になり、次フレームでの予測精度が向上します。
 * @param mean 平均（予測値を入力して、補正値が出力されます。）
 * @param covariance 共分散（予測値を入力して、補正値が出力されます。）
 * @param measurement 観測されたbounding box [bbox center x, bbox center y, aspect, height]
 */
void byte_track::KalmanFilter::update(StateMean &mean, StateCov &covariance, const DetectBox &measurement) {
    StateHMean projected_mean; // 4次元の予測観測値
    StateHCov projected_cov; // 4×4の予測観測共分散
    project(projected_mean, projected_cov, mean, covariance);

    // Kalman Gainの計算
    // B = P * H^T の転置
    const Eigen::Matrix<float, 4, 8> B = (covariance * (update_mat_.transpose())).transpose();
    // kalman_gain = (S^(-1) * H * P)^T = P * H^T * S^(-1)
    Eigen::Matrix<float, 8, 4> kalman_gain = (projected_cov.llt().solve(B)).transpose();

    // Innovation（実際の観測値と予測観測値の差）の計算
    const Eigen::Matrix<float, 1, 4> innovation = measurement - projected_mean;

    // 状態の更新
    // tmp = innovation * K^T = y * K^T
    const auto tmp = innovation * (kalman_gain.transpose());
    // mean_new = mean_old + K * y
    mean = (mean.array() + tmp.array()).matrix();
    // P_new = P_old - K * S * K^T
    covariance = covariance - kalman_gain * projected_cov * (kalman_gain.transpose());
}

/**
 * 状態空間から観測空間への射影（projection）を行う処理です。
 * mean(3)はバウンディングボックスの高さを表し、オブジェクトサイズに比例した観測ノイズを設定
 * @param projected_mean 4次元の観測ベクトル [x, y, a, h]
 * @param projected_covariance 4x4の観測共分散
 * @param mean 平均（予測値）
 * @param covariance 共分散（予測値）
 */
void byte_track::KalmanFilter::project(StateHMean &projected_mean, StateHCov &projected_covariance,
                                       const StateMean &mean, const StateCov &covariance) {
    // 観測ノイズの標準偏差
    DetectBox std;
    std << std_weight_position_ * mean(3), // x座標の観測ノイズ
            std_weight_position_ * mean(3), // y座標の観測ノイズ
            1e-1, // アスペクト比の観測ノイズ
            std_weight_position_ * mean(3); // 高さの観測ノイズ

    projected_mean = update_mat_ * mean.transpose();

    // 共分散の線形変換
    projected_covariance = update_mat_ * covariance * (update_mat_.transpose());

    // 観測ノイズ共分散行列を追加
    Eigen::Matrix<float, 4, 4> diag = std.asDiagonal();
    projected_covariance += diag.array().square().matrix();
}
