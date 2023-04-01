# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
import torch
import csv
import os

from aimet_torch import utils
from aimet_torch.meta.operation import Op

from aimet_torch.arch_checker.arch_checker import ArchChecker
from aimet_torch.arch_checker.arch_checker_rules import TorchActivations
from aimet_torch.arch_checker.constants import ArchCheckerReportConstants as report_const



class Model(torch.nn.Module):
    """
    Model that uses functional modules instead of nn.Modules.
    Expects input of shape (1, 3, 32, 32)
    """
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(10, 32, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.relu1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0, bias=False)
        self.relu2 = torch.nn.ReLU()

        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 48, kernel_size=2, stride=2, padding=2, bias=False)

        self.conv4 = torch.nn.Conv2d(48, 20, 3)
        self.bn3 = torch.nn.BatchNorm2d(20)
        self.bn4 = torch.nn.BatchNorm2d(20)

        self.fc1 = torch.nn.Linear(1280, 10)
        self.prelu = torch.nn.PReLU()
        self.silu = torch.nn.SiLU()

    def forward(self, x):
        # Regular case - conv followed by bn
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Non-linearity between conv and bn, not a candidate for fold
        x = self.conv2(x)
        x = self.relu2(x)

        # Case where BN can fold into an immediate downstream conv
        x = self.bn2(x)
        x = self.conv3(x)

        # No fold if there is a split between conv and BN
        x = self.conv4(x)
        bn1_out = self.bn3(x)
        bn2_out = self.bn4(x)

        x = bn1_out + bn2_out

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.prelu(x)
        x = self.silu(x)
        return x

class TestArchChecker():
    """ Class for testing arch (architechture) checker. """
    model = Model()
    dummy_input = utils.create_rand_tensors_given_shapes((2, 10, 64, 64), utils.get_device(model))

    def test_arch_checker_report(self):
        """ Test exported functions in ArchCheckerReport Class. """
        def read_csv(file_path):
            csv_file = []
            with open(file_path) as file:
                f = csv.reader(file, delimiter='\t')
                for line in f:
                    csv_file.append(line)
            return csv_file

        def get_export_dict(csv_file):
            # Remove header
            csv_file = csv_file[1:]
            export_dict = {}
            for line in csv_file:
                _, module_name, issue, recomm = line
                if module_name not in export_dict:
                    export_dict[module_name] = {report_const.DF_ISSUE: {issue},
                                                report_const.DF_RECOMM: {recomm}}
                else:
                    export_dict[module_name][report_const.DF_ISSUE].update({issue})
                    export_dict[module_name][report_const.DF_RECOMM].update({recomm})
            return export_dict

        arch_checker_report = ArchChecker.check_model_arch(self.model, self.dummy_input)
        export_path = arch_checker_report.export_path
        
        # Add undefined check results to raw result.
        test_op = Op(name="test_op", dotted_name="test_dotted_name", output_shape =None, 
                 is_anonymous=False, op_type="test_type", residing_module=None)
        unknown_check_name = "unknow_check"

        arch_checker_report.raw_report["Model.conv1"].add_failed_checks({unknown_check_name})
        arch_checker_report.update_raw_report(test_op, {unknown_check_name} ) 

        arch_checker_report.export_checker_report_to_cvs()

        # Read export csv file.
        csv_file = read_csv(export_path)
        export_dict = get_export_dict(csv_file)

        # unknown_check_name raises undefined message.
        assert report_const.UNDEFINED_ISSUE.format(unknown_check_name) in export_dict[test_op.dotted_name][report_const.DF_ISSUE]
        assert report_const.UNDEFINED_RECOMM.format(unknown_check_name) in export_dict[test_op.dotted_name][report_const.DF_RECOMM]

        assert report_const.UNDEFINED_ISSUE.format(unknown_check_name) in export_dict["Model.conv1"][report_const.DF_ISSUE]
        assert report_const.UNDEFINED_RECOMM.format(unknown_check_name) in export_dict["Model.conv1"][report_const.DF_RECOMM]

        os.remove(export_path)
        assert not os.path.exists(export_path)

        # Export to pass-in path
        test_export_path = "test_export_path.csv"
        arch_checker_report.export_checker_report_to_cvs(test_export_path)
        
        # Read export csv file.
        csv_file = read_csv(test_export_path)
        export_dict = get_export_dict(csv_file)

        # unknown_check_name raises undefined message.
        assert report_const.UNDEFINED_ISSUE.format(unknown_check_name) in export_dict[test_op.dotted_name][report_const.DF_ISSUE]
        assert report_const.UNDEFINED_RECOMM.format(unknown_check_name) in export_dict[test_op.dotted_name][report_const.DF_RECOMM]

        assert report_const.UNDEFINED_ISSUE.format(unknown_check_name) in export_dict["Model.conv1"][report_const.DF_ISSUE]
        assert report_const.UNDEFINED_RECOMM.format(unknown_check_name) in export_dict["Model.conv1"][report_const.DF_RECOMM]

        os.remove(test_export_path)
        assert not os.path.exists(test_export_path)

        # Test setter fot export_path.
        new_export_path = "./new_export_path.csv"
        arch_checker_report.export_path = new_export_path

        # Test reset_raw_report. 
        arch_checker_report.reset_raw_report()
        arch_checker_report.export_checker_report_to_cvs()
        assert os.path.exists(new_export_path)

        csv_file = read_csv(new_export_path)
        # An empty report should export header only.
        assert len(csv_file) == 1
        assert csv_file[0][1:] == report_const.OUTPUT_CSV_HEADER
        os.remove(new_export_path)
        

    def test_check_arch(self):
        """ Test check_arch function with self defined model."""
        arch_checker_report = ArchChecker.check_model_arch(self.model, self.dummy_input)
        # Node check unit test
        # Model.conv1 has input channel = 3, should fail _check_conv_channel_32_base and
        # _check_conv_channel_larger_than_32
        assert "_check_conv_channel_32_base" in arch_checker_report.raw_report['Model.conv1'].failed_checks
        assert "_check_conv_channel_larger_than_32" in arch_checker_report.raw_report['Model.conv1'].failed_checks

        # Model.conv2 should pass all the checks. No return.
        assert 'Model.conv2' not in arch_checker_report.raw_report

        # Model.conv3 has output channel = 48. should fail _check_conv_channel_32_base
        assert "_check_conv_channel_32_base" in arch_checker_report.raw_report['Model.conv3'].failed_checks

        # prelu and silu should not pass not prelu check.
        assert "_activation_checks" in arch_checker_report.raw_report['Model.prelu'].failed_checks
        assert "_activation_checks" in arch_checker_report.raw_report['Model.silu'].failed_checks

        # relu should pass all checks
        assert "Model.relu1" not in arch_checker_report.raw_report
        assert "Model.relu2" not in arch_checker_report.raw_report

        # Pattern check unit test
        # bn1 can be folded into conv1
        assert "_check_batch_norm_fold" not in arch_checker_report.raw_report

        # bn2 can be folded into conv3
        assert "_check_batch_norm_fold" not in arch_checker_report.raw_report

        # bn3 and bn4 has a split between conv4, can not be folded
        assert "_check_batch_norm_fold" in arch_checker_report.raw_report['Model.bn3'].failed_checks
        assert "_check_batch_norm_fold" in arch_checker_report.raw_report['Model.bn4'].failed_checks

    def test_add_node_check(self):
        """
        Test add_check function is arch_checker. Add a test that will always fail: pass if relu is
        conv2d. The added check will always fail to return a failure record.
        """
        def _temp_check_relu_is_conv2d(node)-> bool:
            """ Temp check pass if relu is conv2d. This should always fail. """
            if not isinstance(node, torch.nn.modules.conv.Conv2d):
                return False
            return True
        ArchChecker.add_node_check(torch.nn.ReLU, _temp_check_relu_is_conv2d)

        arch_checker_report = ArchChecker.check_model_arch(self.model, self.dummy_input)

        # Relu is TorchActivations. Should under TorchActivations checks.
        assert torch.nn.ReLU not in ArchChecker._node_check_dict

        # _temp_check_relu_is_conv2d subject to Relu(TorchActivations) same func.__name__.
        assert _temp_check_relu_is_conv2d.__name__ in [_check.__name__ for _check in ArchChecker._node_check_dict[TorchActivations]]

        # _temp_check_relu_is_conv2d subject to Relu(TorchActivations). prelu and swish should node should return True without being checked.
        assert _temp_check_relu_is_conv2d.__name__ not in arch_checker_report.raw_report['Model.prelu'].failed_checks
        assert _temp_check_relu_is_conv2d.__name__ not in arch_checker_report.raw_report['Model.silu'].failed_checks

        # 'relu1'node is ReLU not Conv2d, so failed the _relu_is_Conv2d test.
        assert _temp_check_relu_is_conv2d.__name__ in arch_checker_report.raw_report['Model.relu1'].failed_checks
        assert "_activation_checks" not in arch_checker_report.raw_report['Model.relu1'].failed_checks

    def test_add_pattern_check(self):
        """
        Test add_check function is arch_checker. Add a test that will always fail: pass if relu is
        conv2d. The added check will always fail to return a failure record.
        """
        def _temp_check_get_all_bns(connected_graph):
            """ Temp check pass if relu is conv2d. This should always fail. """
            _bn_linear_optypes = ['BatchNormalization', 'BatchNorm3d']
            bn_ops = [op for op in connected_graph.get_all_ops().values() if op.type in _bn_linear_optypes]
            return bn_ops

        ArchChecker.add_pattern_check(_temp_check_get_all_bns)

        arch_checker_report = ArchChecker.check_model_arch(self.model, self.dummy_input)

        # all bns should be listed
        assert _temp_check_get_all_bns.__name__ in arch_checker_report.raw_report['Model.bn1'].failed_checks
        assert _temp_check_get_all_bns.__name__ in arch_checker_report.raw_report['Model.bn2'].failed_checks
        assert _temp_check_get_all_bns.__name__ in arch_checker_report.raw_report['Model.bn3'].failed_checks
        assert _temp_check_get_all_bns.__name__ in arch_checker_report.raw_report['Model.bn4'].failed_checks
