#include "ByteTrack/LabeledBYTETracker.h"
#include "ByteTrack/STrack.h"

byte_track::LabeledSTrack::LabeledSTrack(const std::shared_ptr<STrack> &s_track, const int &label): s_track_(s_track),
    label_(label) {
}

byte_track::LabeledSTrack::~LabeledSTrack() = default;

const std::shared_ptr<byte_track::STrack> &byte_track::LabeledSTrack::getSTrack() const {
    return s_track_;
}

const int &byte_track::LabeledSTrack::getLabel() const {
    return label_;
}

byte_track::LabeledBYTETracker::LabeledBYTETracker(const int &frame_rate, const int &track_buffer,
                                                   const float &track_thresh, const float &high_thresh,
                                                   const float &match_thresh): frame_rate_(frame_rate),
                                                                               track_buffer_(track_buffer),
                                                                               track_thresh_(track_thresh),
                                                                               high_thresh_(high_thresh),
                                                                               match_thresh_(match_thresh) {
}

byte_track::LabeledBYTETracker::~LabeledBYTETracker() = default;

std::vector<byte_track::LabeledSTrack> byte_track::LabeledBYTETracker::update(const std::vector<Object> &objects) {
    std::vector<LabeledSTrack> results;
    std::map<int, std::vector<Object> > objects_map;
    for (const auto &object: objects) {
        objects_map[object.label].push_back(object);
    }
    for (const auto &[label, object_list]: objects_map) {
        if (labeled_trackers_.count(label) < 1) {
            std::unique_ptr<BYTETracker> ptr_tracker(
                new BYTETracker(frame_rate_, track_buffer_, track_thresh_, high_thresh_, match_thresh_));
            labeled_trackers_[label] = std::move(ptr_tracker);
        }
        const auto updated_per_label = labeled_trackers_[label]->update(object_list);
        for (const auto &ret: updated_per_label) {
            results.emplace_back(ret, label);
        }
    }
    return results;
}
