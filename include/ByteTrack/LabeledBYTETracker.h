#pragma once

#include "ByteTrack/STrack.h"
#include "ByteTrack/BYTETracker.h"

namespace byte_track {
    class LabeledSTrack {
    public:
        LabeledSTrack(const std::shared_ptr<STrack> &s_track, const int &label);

        ~LabeledSTrack();

        const std::shared_ptr<STrack> &getSTrack() const;

        const int &getLabel() const;

    private:
        std::shared_ptr<STrack> s_track_;
        int label_;
    };

    class LabeledBYTETracker {
    public:
        using BYTETrackerPtr = std::unique_ptr<BYTETracker>;
        using STrackPtr = std::shared_ptr<STrack>;

        LabeledBYTETracker(const int &frame_rate = 30,
                           const int &track_buffer = 30,
                           const float &track_thresh = 0.5,
                           const float &high_thresh = 0.6,
                           const float &match_thresh = 0.8);

        ~LabeledBYTETracker();

        std::vector<LabeledSTrack> update(const std::vector<Object> &objects);

    private:
        const int frame_rate_;
        const int track_buffer_;
        const float track_thresh_;
        const float high_thresh_;
        const float match_thresh_;

        std::map<int, BYTETrackerPtr> labeled_trackers_;
    };
}
